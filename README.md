# ConPer

Code and datasets for the paper [Persona-Guided Planning for Persona-Aware Story Generation](https://arxiv.org/pdf/2204.10703.pdf)

## 1. Environment Setup

All experiments were conducted on a single NVIDIA GPU

Use conda for pakage compatity

For GPT2-based ConPer 
use conda create --name myenv python=3.7
Install dependencies
```
pip install -r requirements_gpt2.txt
```

For BART-based ConPer 
use conda create --name myenv python=3.8
Install dependencies
```
pip install -r requirements_bart.txt
```
## 2.Run
### Preparation

#### Download datasets
The preprocessed data is in `data/`.
`small_test_data.json` is the test set for the original paper's pretrained model.
Files with name like `.._subset.json` are splited from the original training set to train gpt2 and bart based model from start with small size data.

#### Download fine-tuned model

The original fine-tuned model is `results/ckpt/epoch=4-step=29944.ckpt`.

The trained gpt2-based model by small data size is `results/ckpt/epoch=2-step=1198.ckpt`.

The trained bart-based model by small data size is `bart-epochepoch=03-val_lossval_loss=3.7787.ckpt`.

### Train

To train a model, you can run the following command, where `0` denotes GPU_ID.

gpt2-based
```
python src/main.py --train_data data/train_subset.json --valid_data data/valid_subset.json --config_path gpt2 --epoch 3 --batch_size 4 --accumulate_grad 2 --gpu 0 --save_dir results
```

bart-based
```
python src/main2.py --train_data data/train_subset.json --valid_data data/valid_subset.json --config_path facebook/bart-large --epoch 3 --batch_size 4 --accumulate_grad 2 --gpu 0 --save_dir results
```
You can replace with larger dataset if resources approved.

### Generate

To generate stories, you can run the following command, where `0` denotes GPU_ID.

```
python src/main.py --generate --test_data data/small_test_data.json --output_path results/lightning_logs/gen.json  --ckpt_path results/ckpt/epoch=4-step=29944.ckpt --config_path gpt2 --batch_size 4 --accumulate_grad 8 --gpu 0
```

### Evaluation

Evaluation files are in `src/evaluation/`

Run if you need to evaluate the results.



