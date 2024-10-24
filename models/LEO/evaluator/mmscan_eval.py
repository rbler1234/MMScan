### use the embodiedscan eval here
import importlib
import sys
sys.path.append('/mnt/petrelfs/linjingli/MMScan_code/embodiedscan')
MMScan_eval = importlib.import_module('eval.tradition_evaluation')

model_config = {"simcse":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/pc","sbert":"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/st"}


## TODO: maybe you need to multi gpu step

class MMScanEvaluator(MMScan_eval.Tradition_Evaluator):
    def __init__(self, eval_bs,model_config = model_config,max_length=1024, verbose=False):
        super(MMScanEvaluator, self).__init__(eval_bs,model_config,max_length,verbose)
        self.target_metric = "refined_EM"
        self.best_result = 0.0
    
    def to_mmscan_form(self,raw_batch_result):
         
        batch_result = {}
        for ID, pred, gt, instruction in zip(raw_batch_result['question_id'],raw_batch_result['output_txt'],raw_batch_result['output_gt'],raw_batch_result['prompt_after_obj']):
            batch_result[ID] = {'pred':pred,
                                  'gt':gt,
                                  'instruction':instruction,
                                  'question':instruction
                                  }

        return batch_result
        

    def record(self, split, is_main_process):
        # record after a whole epoch
        self.start_evaluation()
        print(self.metric_record)
        score = self.metric_record[self.target_metric]
        
        if score > self.best_result:
            is_best = True
            self.best_result = score
        else:
            is_best = False
        if is_main_process:
            print('An epoch is over!')
   
        return is_best, self.metric_record
    
if __name__=="__main__":
    
    eval_bs = 400
    
    test_evaluator = MMScanEvaluator(eval_bs,{},verbose=True)
    
    # import json
    # with open('/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/test_leo_new/leo_test_pred_gt.json','r') as f:
    #     data_dict = json.load(f)
    # print(len(data_dict))
    # ex_output1 = data_dict
    
    ex_output1 = {'question_id':[f'question_1',f'question_2',f'question_3',f'question_4'],
                 'output_txt':[['Yes, it is']]*4,
                 'output_gt':[['Yes, there is.']]*4,
                 'prompt_after_obj':["Is there is a book in the room?"]*4}

    # ex_output2 = {'question_id':[f'question_5',f'question_6',f'question_7',f'question_8'],
    #              'output_txt':[['Yes, there is.']]*4,
    #              'output_gt':[['Yes, there is.']]*4,
    #              'prompt_after_obj':["Is there is a book in the room?"]*4}

    test_evaluator.update(ex_output1)
    #test_evaluator.update(ex_output2)
    

    _, results = test_evaluator.record(
        split='test', is_main_process=True
    )
    
    print(results)