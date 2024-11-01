# mmscan dev-kit.
# Code written by Jingli Lin, 2024.

from copy import deepcopy
import os
import os.path as osp
import sys
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from mmscan_tool.utils.data_io import (
    read_annotation_pickle,
    id_mapping,
    load_json,
)
from mmscan_tool.utils.box_utils import __9DOF_to_6DOF__
from mmscan_tool.utils.task_utils import anno_token_flatten

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("MMScan dev-kit only supports Python version 3.")

ENV_PATH = os.path.abspath(__file__)


class MMScan(Dataset):
    """Database class for MMScan to help query and retrieve information from
    the database.

    Args:
        version (str): The version of the database, now only support v1.
                        Defaults to 'v1'.
        split (str): The split of the database, now only support train/val.
                        Defaults to 'train'.
        dataroot (str): The root path of the database.
                        Defaults to the path of mmscan_data dir.
        task (str): The language task of the database, now only support
                        MMScan-QA/MMScan-VG.
        ratio (float): The ratio of the data to be used.
                        Defaults to 1.0.
        verbose (bool): Whether to print the information or not.
                        Defaults to False.
        check_mode (bool): Whether to debug or not.
                        Defaults to False.
        token_flatten (bool): It only works in MMScan VG tasks, whether to
                            flatten the tokens or not. Defaults to True.
    """

    def __init__(
        self,
        version: str = "v1",
        split: str = "train",
        dataroot: str = "",
        task: str = None,
        ratio: float = 1.0,
        verbose: bool = False,
        check_mode: bool = False,
        token_flatten: bool = True,
    ):
        """Initialize the database, prepare the embodeidscan annotation."""
        super(MMScan, self).__init__()
        self.version = version
        if len(dataroot) > 0:
            self.dataroot = dataroot
        else:
            self.dataroot = "/mnt/petrelfs/linjingli/mmscan_db/mmscan_data"
            # TODO: change this
            # os.path.join(os.path.dirname(os.path.dirname(ENV_PATH)),'mmscan_data')
        self.verbose = verbose

        # now we skip the test split because we don not provide ground truth for the test split.
        if split == "test":
            split = "val"
        self.split = split
        self.check_mode = check_mode
        if self.check_mode:
            print("embodiedscan's checking mode!!!")
        self.pkl_name = "{}/embodiedscan-split/embodiedscan-{}/embodiedscan_infos_{}.pkl".format(
            self.dataroot, self.version, split
        )
        self.data_path = "{}/embodiedscan-split/data".format(self.dataroot)
        self.lang_anno_path = "{}/MMScan-beta-release".format(self.dataroot)

        self.pcd_path = "{}/embodiedscan-split/process_pcd".format(
            self.dataroot
        )

        self.mapping_json_path = (
            "{}/embodiedscan-split/data_info/mp3d_mapping.json".format(
                self.dataroot
            )
        )
        # TODO: change this
        #'{}/../data_preparation/meta-data/mp3d_mapping.json'.format(self.dataroot)
        self.id_mapping = id_mapping(self.mapping_json_path)
        self.table_names = [
            "es_file",
            "pc_file",
            "point_clouds",
            "bboxes",
            "object_ids",
            "object_types",
            "object_type_ints",
            "visible_view_object_dict",
            "extrinsics_c2w",
            "axis_align_matrix",
            "intrinsics",
            "depth_intrinsics",
            "image_paths",
            "depth_image_paths",
            "visible_instance_ids",
        ]
        self.lang_tasks = ["MMScan-QA", "MMScan-VG", "MMScan-DC"]
        self.task_anno_mapping = {
            "MMScan-QA": "MMScan_QA.json",
            "MMScan-VG": "MMScan_VG.json",
            "MMScan-DC": None,
        }

        # Part1: prepare the embodeidscan annotation.
        assert osp.exists(self.pkl_name), "Database not found: {}".format(
            self.pkl_name
        )
        if verbose:
            start = time.time()
        self.embodiedscan_anno = self.__load_base_anno__(self.pkl_name)
        if verbose:
            print(
                "======\nLoading embodiedscan-{} for split {}, using {} seconds".format(
                    self.version, self.split, time.time() - start
                )
            )

        # Part2: prepare the MMScan annotation.
        self.task = task
        assert (
            self.task_anno_mapping.get(task, None) is not None
        ), "Task {} is not supported yet".format(task)
        if verbose:
            start = time.time()
        self.mmscan_scan_id = load_json(
            f"{self.lang_anno_path}/Data_splits/{self.split}-split.json"
        )
        self.mmscan_anno = load_json(
            f"{self.lang_anno_path}/MMScan_samples/{self.task_anno_mapping[task]}"
        )[self.split]
        if ratio < 1.0:
            self.mmscan_anno = self.__downsample_annos__(
                self.mmscan_anno, ratio
            )
        if self.check_mode:
            self.mmscan_anno = self.mmscan_anno[:100]
        if self.task == "MMScan-VG" and token_flatten:
            self.mmscan_anno = anno_token_flatten(self.mmscan_anno)
            self.mmscan_anno = self.__filter_lang_anno__(self.mmscan_anno)
        self.data_collect()
        if verbose:
            end = time.time()
            print(
                "==================\nLoading {} split for the {} task, using {} seconds.".format(
                    self.split, self.task, end - start
                )
            )

    def __getitem__(self, index_):
        """return the sample item corresponding to the index. The item
        contains: (1) scan-level "ori_pcds" (tuple[tensor]): the raw data read
        from the pth file, contains in order (pcd coordinates, pcd colors, pcd
        class labels, pcd instance labels)

            "pcds" (np.ndarray):
            the point cloud data of the scan,
            [n_points, 6(xyz+rgb)]

            "instance_labels" (np.ndarray):
            the object id of each point,
            [n_points,1]

            "class_labels" (np.ndarray):
            the class type of each point,
            [n_points,1]

            "bboxes" (dict):
            bounding boxes info
            { object_id :
            {"type": object_type (str),
             "bbox": 9 DoF box (np.ndarray),
             ...}

            "images"(list[dict]):
            A list of camera-info.
            [
                {
                'img_path'(str): path to its rgb image
                'depth_img_path'(str): path to its depth image
                'intrinsic'(np.ndarray): camera intrinsic of the rgb image
                'depth_intrinsic'(np.ndarray): camera intrinsic of the depth image
                'extrinsic'(np.ndarray): camera extrinsic
                'visible_instance_id'(list): Ds of objects visible
                }
                ...
            ]

        (2) anno-level (for Visual Grounding task) / (for Question Answering task)

        Args:
            index_ (int): the index
        Returns:
            dict: The sample item corresponding to the index.
        """
        assert self.task is not None, "Please set the task first!"

        # (1) store the "index" info
        data_dict = {}
        data_dict["index"] = index_

        # (2) loading the data
        scan_idx = self.MMScan_collect["anno"][index_]["scan_id"]
        pcd_info = self.__process_pcd_info__(scan_idx)
        images_info = self.__process_img_info__(scan_idx)
        box_info = self.__process_box_info__(scan_idx)

        data_dict["ori_pcds"] = pcd_info["ori_pcds"]
        data_dict["pcds"] = pcd_info["pcds"]
        data_dict["obj_pcds"] = pcd_info["obj_pcds"]
        data_dict["instance_labels"] = pcd_info["instance_labels"]
        data_dict["class_labels"] = pcd_info["class_labels"]
        data_dict["bboxes"] = box_info
        data_dict["images"] = images_info

        # (3) loading the data from the collection
        # necessary to use deepcopy?
        data_dict.update(deepcopy(self.MMScan_collect["anno"][index_]))

        return data_dict

    def __len__(self):
        assert self.task is not None, "Please set the task first!"
        return len(self.MMScan_collect["anno"])

    @property
    def show_possess(self) -> List[str]:
        """
        Returns:
            list[str]: All data classes present in Embodiedscan database.
        """
        return self.table_names

    @property
    def show_mmscan_id(self) -> List[str]:
        """
        Returns:
            list[str]: All data classes present in Embodiedscan database.
        """
        assert self.task is not None, "Please set the task first!"
        return self.mmscan_scan_id

    @property
    def samples(self):
        """
        Returns:
            list[dict]: All samples in the MMScan language task.
        """
        assert self.task is not None, "Please set the task first!"
        return self.MMScan_collect["anno"]

    def get_possess(self, table_name: str, scan_idx: str):
        """Getting all database about the scan from embodeidscan.

        Args:
            table_name (str): type of the expected data.
            scan_idx (str): The scan id to get the data.
        Returns:
            The data corresponding to the table_name and scan_idx.
        """
        assert table_name in self.table_names, "Table {} not found".format(
            table_name
        )

        if table_name == "point_clouds":
            return torch.load(
                f"{self.pcd_path}/{self.id_mapping.forward(scan_idx)}.pth"
            )
        elif table_name == "es_file":
            return deepcopy(self.pkl_name)
        elif table_name == "pc_file":
            return f"{self.pcd_path}/{self.id_mapping.forward(scan_idx)}.pth"
        else:
            return self.embodiedscan_anno[scan_idx][table_name]

    def data_collect(self) -> dict:
        """Collecting the MMScan samples.

        Store them in self.MMScan_collect. MMScan QA samples need to be
        flatten.
        """

        assert self.task is not None, "Please set the task first!"
        self.MMScan_collect = {}

        #  MMScan anno processing
        if self.task == "MMScan-QA":
            self.MMScan_collect["anno"] = []

            for sample in self.mmscan_anno:
                if self.split == "train":
                    for answer in sample["answers"]:
                        sub_sample = deepcopy(sample)
                        sub_sample["answers"] = [answer]
                        self.MMScan_collect["anno"].append(sub_sample)
                else:
                    self.MMScan_collect["anno"].append(sample)

        elif self.task == "MMScan-VG":
            self.MMScan_collect["anno"] = self.mmscan_anno
        else:
            raise NotImplementedError

    def __filter_lang_anno__(self, samples):
        """Check and  the annotation is valid or not.

        Args:
            samples (list[dict]): The samples.
        Returns:
            list[dict] : The filtered results.
        """

        if self.task != "MMScan-VG":
            return samples

        filtered_samples = []
        for sample in samples:
            if self.__check_lang_anno__(sample):
                filtered_samples.append(sample)
        return filtered_samples

    def __check_lang_anno__(self, sample) -> bool:
        """Check if the item of the annotation is valid or not.

        Args:
            sample (dict): The item from the samples.
        Returns:
            bool : Whether the item is valid or not.
        """
        if self.task == "MMScan-VG":
            return (
                len(sample["target"]) == len(sample["target_id"])
                and len(sample["target_id"]) > 0
            )

        return True

    def __load_base_anno__(self, pkl_path) -> dict:
        """Load the embodiedscan pkl file, it will return the embodiedscan
        annotations of all scans in the corresponding split.

        Args:
            pkl_path (str): The path of the pkl.
        Returns:
            dict : The embodiedscan annotations of scans.
            (with scan_idx as keys)
        """
        return read_annotation_pickle(pkl_path, show_progress=self.verbose)

    def __process_pcd_info__(self, scan_idx: str):
        """Retrieve the corresponding scan information based on the input scan
        ID, including original data, point clouds, object pointclouds, instance
        labels and the center of the scan.

        Args:
            scan_idx (str): the scan ID.
        Returns:
            dict : corresponding scan information.
        """

        assert (
            scan_idx in self.embodiedscan_anno.keys()
        ), "Scan {} is not in {} split".format(scan_idx, self.split)

        scan_info = {}
        pcd_data = torch.load(
            f"{self.pcd_path}/{self.id_mapping.forward(scan_idx)}.pth"
        )
        points, colors, class_labels, instance_labels = pcd_data

        pcds = np.concatenate([points, colors], 1)
        scan_info["ori_pcds"] = deepcopy(pcd_data)
        scan_info["pcds"] = deepcopy(pcds)

        obj_pcds = {}
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i
            if len(pcds[mask]) > 0:
                obj_pcds.update({i: pcds[mask]})

        scan_info["obj_pcds"] = obj_pcds
        scan_info["scene_center"] = (points.max(0) + points.min(0)) / 2
        scan_info["instance_labels"] = np.array(instance_labels)
        scan_info["class_labels"] = np.array(class_labels)
        return scan_info

    def __process_box_info__(self, scan_idx: str):
        """Retrieve the corresponding bounding boxes information based on the
        input scan ID. For each object, this function will return its ID, type,
        bounding boxes in format of.

        {ID: {"bbox":bbox, "type":type}}

        Args:
            scan_idx (str): the scan ID.
        Returns:
            dict : corresponding bounding boxes
            information.
        """
        assert (
            scan_idx in self.embodiedscan_anno.keys()
        ), "Scan {} is not in {} split".format(scan_idx, self.split)

        bboxes = deepcopy(self.get_possess("bboxes", scan_idx))
        object_ids = deepcopy(self.get_possess("object_ids", scan_idx))
        object_types = deepcopy(self.get_possess("object_types", scan_idx))
        return {
            object_ids[i]: {"bbox": bboxes[i], "type": object_types[i]}
            for i in range(len(object_ids))
        }

    def __process_img_info__(self, scan_idx: str):
        """Retrieve the corresponding camera information based on the input
        scan ID. For each camera, this function will return its intrinsics,
        extrinsics, image paths(both rgb & depth) and the visible object ids.

        Args:
            scan_idx (str): the scan ID.
        Returns:
            list[dict] : corresponding information
            for each camera.
        """
        assert (
            scan_idx in self.embodiedscan_anno.keys()
        ), "Scan {} is not in {} split".format(scan_idx, self.split)

        img_info = dict()
        img_info["img_path"] = deepcopy(
            self.get_possess("image_paths", scan_idx)
        )
        img_info["depth_img_path"] = deepcopy(
            self.get_possess("depth_image_paths", scan_idx)
        )
        img_info["intrinsic"] = deepcopy(
            self.get_possess("intrinsics", scan_idx)
        )
        img_info["depth_intrinsic"] = deepcopy(
            self.get_possess("depth_intrinsics", scan_idx)
        )
        img_info["extrinsic"] = deepcopy(
            self.get_possess("extrinsics_c2w", scan_idx)
        )
        img_info["visible_instance_id"] = deepcopy(
            self.get_possess("visible_instance_ids", scan_idx)
        )

        img_info_list = []
        for camera_index in range(len(img_info["img_path"])):
            item = {}
            for possess in img_info.keys():
                item[possess] = img_info[possess][camera_index]
            img_info_list.append(item)
        return img_info_list

    def down_9DOF_to_6DOF(self, pcd, box_9DOF) -> np.ndarray:
        """Transform the 9DOF bounding box to 6DOF bounding box. Find the
        minimum bounding boxes to cover all the point clouds.

        Args:
            pcd(np.ndarray / Tensor):
                the point clouds
            box_9DOF(np.ndarray / Tensor):
                the 9DOF bounding box
        Returns:
            np.ndarray :
                The transformed 6DOF bounding box.
        """

        return __9DOF_to_6DOF__(pcd, box_9DOF)

    def __downsample_annos__(self, annos: List[dict], ratio: float):
        """downsample the annotations with a given ratio.

        Args:
            annos (list[dict]): the original annotations.
            ratio (float): the ratio to downsample.
        Returns:
            list[dict] : The result.
        """
        d_annos = []
        for index in range(len(annos)):
            if index % int(1 / ratio) == 0:
                d_annos.append(annos[index])
        return d_annos


if __name__ == "__main__":
    test = EmbodiedScan(version="v1", split="val", verbose=True)
    test.set_lang_task("MMScan-VG")
    print(len(test))
    print(test[10])
