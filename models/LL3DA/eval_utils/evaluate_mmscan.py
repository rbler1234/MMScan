import os, sys, time, math, json, importlib
import torch
import datetime
from collections import defaultdict, OrderedDict

from utils.box_util import box3d_iou_batch_tensor
from utils.ap_calculator import APCalculator
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    init_distributed, 
    is_distributed, 
    is_primary, 
    get_rank,
    barrier,
    all_reduce_average,
    all_gather_dict
)
import sys
import importlib
sys.path.append('/mnt/petrelfs/linjingli/MMScan_code/embodiedscan')
MMScan_eval = importlib.import_module('eval.tradition_evaluation')
model_config = {"simcse":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/pc","sbert":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/st"}

class MMScanEvaluator(MMScan_eval.Tradition_Evaluator):
    
    def __init__(self, eval_bs=400,model_config = model_config,max_length=1024, verbose=False):
        super(MMScanEvaluator, self).__init__(eval_bs,model_config,max_length,verbose)
        self.target_metric = "refine_EM"
        self.best_result = 0.0
    
    def to_mmscan_form(self,raw_batch_result):
         
        batch_result = {}
        for ID in raw_batch_result:
            batch_result[ID.split('-')[0]] = {'pred':raw_batch_result[ID]['answer_pred'],
                                                'gt':raw_batch_result[ID]['answer_gt'],
                                                'instruction':ID.split('-')[1],
                                                'question':ID.split('-')[1]
                                                }

        return batch_result


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    
    # prepare ground truth caption labels
    print("preparing corpus...")
    
    #evaluator = MMScanEvaluator()
    
    annotations = dataset_loader.dataset.annotations
    corpus = {
    '-'.join((anno['ID'], anno['question'])): anno['answers'] if 'answers' in anno else anno['caption']  \
            for anno in annotations
    }
    candidates = {}
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    barrier()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask': batch_data_label['qformer_attention_mask'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
        }
        outputs = model(model_input, is_eval=True, task_name='qa')
        
        outputs = dict(
            output_ids=outputs["output_ids"],
        )
        
        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)
        
        output_ids = outputs["output_ids"]  # batch x max_length
        answers = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        sample_index = batch_data_label['scan_idx'].cpu().tolist()
        
        batch_results = {}
        for idx in range(output_ids.shape[0]):
            anno = annotations[sample_index[idx]]
            key = '-'.join((anno['ID'], anno['question']))
            answer = answers[idx]
            answer = ' '.join(filter(lambda w: w, answer.split(' ')))
            batch_results[key] = {'answer_pred': [answer]}
            candidates[key] = answer
            assert key in corpus, f"key {key} not in corpus"
            batch_results[key]['answer_gt'] = corpus[key]
        evaluator.update(batch_results)
      

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)
        
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logout(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                f"Evaluating on iter: {curr_train_iter}; "
                f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
        barrier()
    
    # end of forward pass traversion
    metric_results = evaluator.start_evaluation()
    metric_results_record = evaluator.eval_results_record
    
    if is_primary():
        logout("\n----------------------Evaluation-----------------------\n")
    
        
        with open(os.path.join(args.checkpoint_dir, "corpus_val.json"), "w") as f: 
            json.dump(corpus, f, indent=4)
        
        with open(os.path.join(args.checkpoint_dir, "pred_val.json"), "w") as f:
            json.dump(candidates, f, indent=4)
        
        with open(os.path.join(args.checkpoint_dir, "qa_pred_gt_val.json"), "w") as f:
            pred_gt_val = {}
            for index_, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'instruction': scene_object_id_key.split('-')[1],
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                }
                pred_gt_val[scene_object_id_key].update(metric_results_record[scene_object_id_key.split('-')[0]])
            json.dump(pred_gt_val, f, indent=4)
            json.dump(metric_results, f, indent=4)
    return metric_results