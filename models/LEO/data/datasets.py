import glob
import json
import os
import random

import cv2
import nltk
import numpy as np
import pandas as pd
import pickle
import torch
from accelerate.logging import get_logger
from einops import rearrange
from scipy import sparse
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .data_utils import build_rotate_mat, construct_bbox_corners, convert_pc_to_box, \
                        eval_ref_one_sample, get_sqa_question_type, preprocess_2d
from .eai import CLIPORT_ACTION_SPACE_TOKENIZE, HABITAT_ACTION_SPACE, HABITAT_ACTION_SPACE_TOKENIZE, \
                 _DUMMY_CLIPORT_ACTION, _extract_between, shapenetcore_pp
from .text_pool import *

import importlib
import sys
sys.path.append('/mnt/petrelfs/linjingli/MMScan_code/embodiedscan')
embodiedScan = importlib.import_module('embodiedscan')

logger = get_logger(__name__)

# len(tokenized_sentence) / len(sentence)
LLAMA_TOKEN_SENT_RATIO = 0.24

LEOMIX_REQUIRED_KEYS = [
    'source',
    'prompt_before_obj',
    'prompt_middle_1',
    'prompt_middle_2',
    'prompt_after_obj',
    'obj_fts',
    # 'obj_masks',   # this is filled by dataset wrapper
    'obj_locs',
    'anchor_locs',
    'anchor_orientation',
    'img_fts',   # currently hardcode to 224x224
    'img_masks',
    'output_gt',
]


@DATASET_REGISTRY.register()
class LeoBase(Dataset):
    r""" Unified input format:
    <prompt_before_obj> + <prompt_middle_1> + <img_tokens> + <prompt_middle_2> + <obj_tokens> + <prompt_after_obj>
    <prompt_before_obj>: <role_prompt> + <situation_prompt>
    <prompt_middle_1>: <egoview_prompt> (masked if unnecessary)
    <prompt_middle_2>: <objects_prompt>
    <prompt_after_obj>: <task_prompt>
    <output_gt>: response label, will be appended to input sequence for computing loss during training
    """

    role_prompt = "You are an AI visual assistant situated in a 3D scene. "\
                  "You can perceive (1) an ego-view image (accessible when necessary) and (2) the objects (including yourself) in the scene (always accessible). "\
                  "You should properly respond to the USER's instruction according to the given visual information. "
    situation_prompt = "{situation}"
    egoview_prompt = "Ego-view image:"
    objects_prompt = "Objects (including you) in the scene:"
    task_prompt = "USER: {instruction} ASSISTANT:"

    @staticmethod
    def get_prompts(instruction, situation="", dialogue=None):
        return {
            'prompt_before_obj': LeoBase.role_prompt + LeoBase.situation_prompt.format(situation=situation),
            'prompt_middle_1': LeoBase.egoview_prompt,
            'prompt_middle_2': LeoBase.objects_prompt,
            'prompt_after_obj': LeoBase.task_prompt.format(instruction=instruction) if dialogue is None else dialogue,
        }

    @staticmethod
    def check_output_and_fill_dummy(data_dict):
        if 'anchor_locs' not in data_dict:
            data_dict['anchor_locs'] = torch.zeros(3)
        if 'anchor_orientation' not in data_dict:
            data_dict['anchor_orientation'] = torch.zeros(4)
            data_dict['anchor_orientation'][-1] = 1   # xyzw
        if 'img_fts' not in data_dict:
            data_dict['img_fts'] = torch.zeros(3, 224, 224)   # currently hardcode to 224x224
        if 'img_masks' not in data_dict:
            data_dict['img_masks'] = torch.LongTensor([0]).bool()

        for key in LEOMIX_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
        return data_dict

    def _split_sentence(self, sentence, max_length, prefix=''):
        # only split during training
        if self.split == 'train' and len(prefix + sentence) > max_length:
            all_caps = []
            sents = sentence.split('. ')
            tmp = prefix
            for i in range(len(sents)):
                if len(tmp + sents[i] + '. ') > max_length:
                    all_caps.append(tmp)
                    tmp = prefix
                tmp += sents[i] + '. '

            all_caps.append(tmp)   # last chunk

            # final check
            ret = []
            for cap in all_caps:
                if len(cap) <= max_length:
                    ret.append(cap)
            return ret
        else:
            return [prefix + sentence]


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as array of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, np.split(euler_angles, 3, axis=-1))
    ]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

def is_inside_box(points, center, size, rotation_mat):
    """
        Check if points are inside a 3D bounding box.
        Args:
            points: 3D points, numpy array of shape (n, 3).
            center: center of the box, numpy array of shape (3, ).
            size: size of the box, numpy array of shape (3, ).
            rotation_mat: rotation matrix of the box, numpy array of shape (3, 3).
        Returns:
            Boolean array of shape (n, ) indicating if each point is inside the box.
    """
    assert points.shape[1] == 3, "points should be of shape (n, 3)"
    center = np.array(center) # n, 3
    size = np.array(size) # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (3, 3), f"R should be shape (3,3), but got {rotation_mat.shape}"
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return (pcd_local[:, 0] <= 1) & (pcd_local[:, 1] <= 1) & (pcd_local[:, 2] <= 1)

def pcd_color_transformer(pcd):
    """_
    Transform the color of the point cloud to [-1, 1]
    """
    pcd[:,3:6 ] = pcd[:,3:6]*2 -1
    return pcd



class MMScanDataLoader(embodiedScan.EmbodiedScan):
    
    """
        This is the dataloader for the MMScan-QA dataset.    
    """
    
    def __init__(self, cfg, version='v1', split='train', verbose=True):
        super(self,MMScanDataLoader).__init__(version=version, split=split, verbose=verbose)
        
        self.num_points = cfg.data.embodied_scan_l.num_points
        self.max_obj_len = cfg.data.embodied_scan_l.max_obj_len
        self.max_caption_length = int(cfg.llm.max_out_len / LLAMA_TOKEN_SENT_RATIO)
        self.embodied_scan_info_base = cfg.data.embodied_scan_info_base
        self.cfg = cfg
        if split == 'train':
            self.split = 'train'
            self.pc_type = 'gt'
        else:
            self.split = 'val'
            self.pc_type = getattr(cfg.data.embodied_scan_c, 'pc_type', 'gt')
            
        self.situation_pool = Leo_situation_pool
        self.instruction_pool = Leo_objcap_instruction_pool
        
    def preprocess_pcd(self, obj_pcds, return_anchor=False, rot_aug=True):
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        anchor_loc = None
        for i, obj_pcd in enumerate(obj_pcds):
            try:
                if rot_matrix is not None:
                    obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

                obj_center = obj_pcd[:, :3].mean(0)
                obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_locs.append(np.concatenate([obj_center, obj_size], 0))
                if return_anchor and i == 0:
                    # Select a loc within the obj bbox as the anchor.
                    anchor_loc = obj_pcd[:, :3].min(0) + np.random.rand(3) * obj_size

                # subsample
                pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                            replace=len(obj_pcd) < self.num_points)
                obj_pcd = obj_pcd[pcd_idxs]

                # normalize
                obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
                max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
                if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
                    max_dist = 1
                obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
                obj_fts.append(obj_pcd)
            except Exception as e:
                print(f"{e} ==> skip instance {i}")

        # convert to torch
        try:
            obj_fts = torch.from_numpy(np.stack(obj_fts, 0)).float()
            obj_locs = torch.from_numpy(np.array(obj_locs)).float()
        except Exception as e:
            return self.last_ok

        if return_anchor and anchor_loc is not None:
            anchor_loc = torch.from_numpy(anchor_loc).float()
        else:
            anchor_loc = torch.zeros(3)
        self.last_ok = obj_fts, obj_locs, anchor_loc
        return obj_fts, obj_locs, anchor_loc
    
    def parse_dict(self,data_dict)->dict:
        
        
        scan_id = data_dict["scan_id"]
        ID = data_dict["ID"]
        obj_id = data_dict["input_bboxes_id"]
        
        if self.split == 'train':    
            obj_caption = data_dict["answers"][0]
        else:
            obj_caption = data_dict["answers"]
        input_bboxes = ['input_bboxes']

        obj_pcds = {}
        for object_id in data_dict['obj_pcds'].keys():    
            obj_pcds[object_id] = pcd_color_transformer(data_dict['obj_pcds'][object_id])
        
        scan_pcds = data_dict['pcds']
            
        iou_flag = 1
        if obj_id is not None and len(obj_id)>0 and scan_pcds.shape[0]>0:
            all_obj_mask = []
            
            for input_bbox in input_bboxes:
                bbox = np.array(input_bbox)
                orientation=R.from_euler("zxy",bbox[6:],degrees = False).as_matrix().tolist()
                position=np.array(bbox[:3])
                size=np.array(bbox[3:6])
                all_obj_mask.append(torch.tensor(is_inside_box(scan_pcds[:,:3],position,size,orientation),dtype=bool))
            query_instance_mask = torch.stack(all_obj_mask)
            query_instance_mask = torch.any(query_instance_mask, dim=0)
            if query_instance_mask.numel() >0:
                selected_obj_pcds = [scan_pcds[query_instance_mask] ]   
            else:
                print("warning: unable to match pcd inside bbox")
                selected_obj_pcds = []
            
        
            selected_obj_pcds = []
        else:
            selected_obj_pcds = []
        remained_obj_idx = [i for i in obj_pcds.keys()] 

        num_selected_obj = len(selected_obj_pcds)
        if num_selected_obj >= self.max_obj_len:
            selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
        else:
            if self.split == 'train':
                random.shuffle(remained_obj_idx)
            selected_obj_pcds.extend(
                [obj_pcds[i] for i in remained_obj_idx[: self.max_obj_len - num_selected_obj]]
            )
        try:
            obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=obj_id is not None)
        except Exception as e:
            print(scan_id,len(selected_obj_pcds))
            obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=obj_id is not None)
        parse_data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_pool),
            situation=random.choice(self.situation_pool) if obj_id is not None else None,
            dialogue=data_dict['question']
        )
     
        parse_data_dict.update({
            'source': 'scannet', # default 
            'scene_id': scan_id,
            'question_id':ID,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224), # default for LEO
            'img_masks': torch.LongTensor([0]).bool(), # default for LEO
            'output_gt': obj_caption,
            'iou_flag': torch.LongTensor([iou_flag]).bool(),
        })
        
        return parse_data_dict   
    

@DATASET_REGISTRY.register()
class LeoEmbodiedScanL(LeoBase):
    

    def __init__(self, cfg, split):
        super().__init__()
        self.dataloader = MMScanDataLoader(cfg=cfg,version='v1',split=split,verbose=True)
        self.dataloader.set_lang_task("MMScan-QA")

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, index):
        # prepare the data from MMScan api
        data_dict = self.dataloader[index]
        
        return self.check_output_and_fill_dummy(data_dict)



@DATASET_REGISTRY.register()
class LeoMix(Dataset):
    mapping = {
        'embodied_scan_l':LeoEmbodiedScanL,
    }

    def __init__(self, cfg, split):
        self.datasets = []
        self.ratio = cfg.task.leomix.ratio
        logger.info(f"LeoMix about to load: {cfg.task.leomix.mix}")
        for dataset in cfg.task.leomix.mix:
            self.datasets.append(self.mapping[dataset](cfg, split))

        if type(self.ratio) == int or type(self.ratio) == float:
            self.index_range = list(np.cumsum([int(len(d)*self.ratio) for d in self.datasets]))
        else:
            self.index_range = list(np.cumsum([int(len(d)*self.ratio[i]) for i, d in enumerate(self.datasets)]))
        self.index_range = [0] + self.index_range
        logger.info(f"Indices of LeoMix datasets: {self.index_range}")

    def __len__(self):
        return self.index_range[-1]

    @staticmethod
    def streamline_output(data_dict):
        new_data_dict = {}
        for key in LEOMIX_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
            else:
                new_data_dict[key] = data_dict[key]
        return new_data_dict

    def __getitem__(self, index):
        for i in range(len(self.index_range)-1):
            if self.index_range[i] <= index < self.index_range[i+1]:
                data_dict = self.datasets[i][index-self.index_range[i]]
                break

        return self.streamline_output(data_dict)
    
if __name__=="__main__":
    test_leo = LeoEmbodiedScanL(None,"val")
    print(test_leo[1])