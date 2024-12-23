## Visual Grounding Models

### Scanrefer

1. Follow the [Scanrefer](https://github.com/daveredrum/ScanRefer/blob/master/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to download the [preprocessed GLoVE embeddings](https://kaldir.vc.in.tum.de/glove.p) (~990MB) and put them under `data/`

2. Install MMScan API.

3. Overwrite the `lib/config.py/CONF.PATH.OUTPUT` to your desired output directory.

4. Run the following command to train Scanrefer (one GPU):
    ```bash
    python -u scripts/train.py --use_color --epoch {10/25/50}
    ```
5. Run the following command to evaluate Scanrefer (one GPU):
    ```bash
    python -u scripts/train.py --use_color --eval_only --use_checkpoint "path/to/pth"
    ```
### EmbodiedScan
1. Follow the [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/README.md) to setup the Env. You need not load the datasets!

2. Install MMScan API.


3. Run the following command to train Scanrefer (multiple GPU):
    ```bash
    # Single GPU training
    python tools/train.py configs/grounding/pcd_vg_mmscan.py --work-dir=path/to/save

    # Multiple GPU training
    python tools/train.py configs/grounding/pcd_vg_mmscan.py --work-dir=path/to/save --launcher="pytorch"
    ```
5. Run the following command to evaluate Scanrefer (multiple GPU):
    ```bash
    # Single GPU testing
    python tools/test.py configs/grounding/pcd_vg_mmscan.py path/to/load_pth

    # Multiple GPU testing
    python tools/test.py configs/grounding/pcd_vg_mmscan.py path/to/load_pth --launcher="pytorch"
    ```
## Question Answering Models

### LL3DA

1. Follow the [LL3DA](https://github.com/Open3DA/LL3DA/blob/main/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to:

    (1) download the [release pre-trained weights.](https://huggingface.co/CH3COOK/LL3DA-weight-release/blob/main/ll3da-opt-1.3b.pth) and put them under `./pretrained`

    (2) Download the [pre-processed BERT embedding weights](https://huggingface.co/CH3COOK/bert-base-embedding/tree/main) and store them under the `./bert-base-embedding` folder

2. Install MMScan API.

3. Edit the config under `./scripts/opt-1.3b/eval.mmscanqa.sh` and `./scripts/opt-1.3b/tuning.mmscanqa.sh`

4. Run the following command to train LL3DA (4 GPU):
    ```bash
    bash scripts/opt-1.3b/tuning.mmscanqa.sh     
    ```
5. Run the following command to evaluate LL3DA (4 GPU):
    ```bash
    bash scripts/opt-1.3b/eval.mmscanqa.sh 
    ```
    Optinal: You can use the GPT evaluator by this after getting the result.
     'qa_pred_gt_val.json' will be generated under the checkpoint folder after evaluation and the tmp_path is used for temporarily storing.
    ```bash
    python eval_utils/evaluate_gpt.py --file path/to/qa_pred_gt_val.json
    --tmp_path path/to/tmp  --api_key your_api_key --eval_size -1
    --nproc 4

### LEO

1. Follow the [LEO](https://github.com/embodied-generalist/embodied-generalist/blob/main/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to:

    (1) Download [Vicuna-7B](https://huggingface.co/huangjy-pku/vicuna-7b/tree/main) and update cfg_path in configs/llm/*.yaml

    (2) Download the [sft_noact.pth](https://huggingface.co/datasets/huangjy-pku/LEO_data/tree/main) and store it under the `./weights` folder

2. Install MMScan API.

3. Edit the config under `scripts/train_tuning_mmscan.sh` and `scripts/test_tuning_mmscan.sh`

4. Run the following command to train LEO (4 GPU):
    ```bash
    bash scripts/train_tuning_mmscan.sh  
    ```
5. Run the following command to evaluate LEO (4 GPU):
    ```bash
    bash scripts/test_tuning_mmscan.sh
    ```
    Optinal: You can use the GPT evaluator by this after getting the result.
     'test_embodied_scan_l_complete.json' will be generated under the checkpoint folder after evaluation and the tmp_path is used for temporarily storing.
    ```bash
    python evaluator/GPT_eval.py --file path/to/test_embodied_scan_l_complete.json
    --tmp_path path/to/tmp  --api_key your_api_key --eval_size -1
    --nproc 4

PS : It is possible that LEO may encounter an "NaN" error in the MultiHeadAttentionSpatial module due to the training setup when training more epoches. ( no problem for 4GPU one epoch)