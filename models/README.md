## Visual Grounding models

### Scanrefer

1. Follow the [Scanrefer](https://github.com/daveredrum/ScanRefer/blob/master/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to download the [preprocessed GLoVE embeddings](https://kaldir.vc.in.tum.de/glove.p) (~990MB) and put them under `data/`

2. Install MMScan API.

3. Overwrite the `lib/config.py/CONF.PATH.OUTPUT` to your desired output directory.

4. Run the following command to train Scanrefer (one GPU):
    ```bash
    python -u scripts/train.py --use_color --epoch {20/50/100}
    ```
5. Run the following command to evaluate Scanrefer (one GPU):
    ```bash
    python -u scripts/train.py --use_color --eval_only --use_checkpoint "path/to/pth"
    ```
### Embodiedscan
TBD
## Question Answering models

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

TBD