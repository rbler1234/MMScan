import json
from embodiedscan.eval.tradition_evaluation import Tradition_Evaluator

def change_form(data):
    new_dict=dict()
    for item_key in data.keys():
        
        index_ = item_key.index('-')
        ID_,question = item_key[:index_],item_key[index_+1:]
        
        new_dict[ID_] = dict()
        new_dict[ID_]["question"] = question
        new_dict[ID_]["pred"] = data[item_key]["pred"]
        new_dict[ID_]["gt"] = data[item_key]["gt"][0]
    return new_dict

if __name__ == "__main__":
    ex_json_path = '/mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LL3DA_new/ckpts/opt-1.3b/ll3da-mmscan-tuned-test2/embodiedscan_L_pred_gt_val_200.json'

    with open(ex_json_path,'r') as f:
        result_dict = json.load(f)
    result_dict = {key_:result_dict[key_] for key_ in list(result_dict.keys())[:50]}
    result_dict = change_form(result_dict)
    print(len(result_dict))
    evaluator = Tradition_Evaluator(eval_bs = 50 ,model_config ={}, verbose=True)
    print(evaluator.load_and_eval(result_dict))
    
    
