# ELLE: Efficient Lifelong Pre-training for Emerging Data

Code associated with the ELLE: Efficient Lifelong Pre-training for Emerging Data ACL 2020 paper

## Citation

## Installation

```bash
conda env create -f environment.yml
conda activate ELLE
cd ./fairseq_ELLE
pip3 install --editable ./
cd ../apex
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Available Pre-trained Models

We've prepared [pre-trained checkpoints](https://drive.google.com/drive/folders/1aHAdnXm0s3WU-JUc2CXiSA0O3qnJtfwn?usp=sharing) that takes $\text{BERT}_\text{L6_D384}$ as the initial model in fairseq and huggingface formats. 



## Fine-tune

### Downloading Pre-trained Checkpoints

First download the pre-trained checkpoints. 

### Downloading data

#### MNLI 

Follow `/fairseq-0.9.0/README.glue.md` to download and pre-process MNLI dataset, and place it under `./fairseq-0.9.0`.  The directory is expected to be in the structure below:

```
.
| - downstream
| - fairseq_ELLE
| - fairseq-0.9.0
| - - MNLI-bin
| - checkpoints_hf
| - - roberta_base_ELLE
| - checkpoints_fairseq
| - - roberta_base_ELLE

```

#### HyperPartisan, Helpfulness, ChemProt, and ACL-ARC

All these task data is available on a public S3 url; check `./downstream/environments/datasets.py`.

If you run the `./downstream/train_batch.py` command (see next step), we will automatically download the relevant dataset(s) using the URLs in ``./downstream/environments/datasets.py``.

### Fine-tune

#### MNLI

```bash
export PYTHONPATH=./fairseq-0.9.0
cd ./fairseq-0.9.0
bash eval_MNLI_base_prompt.sh
```

#### HyperPartisan

```bash
cd ./downstream
bash finetune_news.sh
```

#### Helpfulness

```bash
cd ./downstream
bash finetune_reviews.sh
```

#### ChemProt

```bash
cd ./downstream
bash finetune_bio.sh
```

#### ACL-ARC

```bash
cd ./downstream
bash finetune_cs.sh
```



## Pre-training

### Prepare Datasets

The dataset of WB domain follows https://arxiv.org/abs/2105.13880 and datasets of News, Review, Bio, CS domains follow https://github.com/allenai/dont-stop-pretraining.  You also need to cut a part of training dataset as the memory. In our main experiment, we take 1G data per domain as the memory.

### Pre-training with ELLE

Firstly, install the fairseq package:

```
export PYTHONPATH=./fairseq_ELLE
cd ./fairseq_ELLE/examples/roberta/
```

Pre-train PLMs with ELLE that takes $\text{BERT}_\text{L6_D384}$ the initial model:

```
bash train_base_prompt.sh 
```

Pre-train PLMs with ELLE that takes $\text{BERT}_\text{L12_D768}$ as the initial model:

```
bash train_large_prompt.sh 
```

Pre-train PLMs with ELLE that takes $\text{GPT}_\text{L6_D384}$ as the initial model:

```
bash gpt_base_prompt.sh 
```

Note that you need to replace the  **DATA_DIR** and **memory_dir** variables in these bash files with your own path to data files and memory files.



### Convert Fairseq Checkpoints into Huggingface Format

Firstly, you need to organize your fairseq PLM checkpoint like the following: 

```
checkpoints_fairseq_new/roberta_base_ELLE/checkpoint_last.pt
```

and copy the dictionary file:

```
cp /downstream/dict.txt /checkpoints_fairseq_new/roberta_base_ELLE
```

Then convert the checkpoint into huggingface checkpoint:

```
cd /downstream
python convert_pnn_to_hf_batch.py /checkpoints_fairseq_new /checkpoints_hf_new
cp -r /downstream/base_prompt_files/* /checkpoints_hf_new
```

Then you can do fine-tuning as **Fine-tune** Section.

