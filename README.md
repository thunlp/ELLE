# Readme

### Prepare Datasets

The dataset of WB domain follows https://arxiv.org/abs/2105.13880 and datasets of News, Review, Bio, CS domains follow https://github.com/allenai/dont-stop-pretraining.  You also need to cut a part of training dataset as the memory.

### Pre-training with ELLE

Firstly, install the fairseq package:

```
cd /fairseq_ELLE
pip install --editable ./
cd /fairseq_ELLE/examples/roberta/
export PYTHONPATH=/fairseq-ELLE
```

Pre-train PLMs with ELLE that takes $\text{BERT}_\text{L6_D384}$ as the initial model:

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



### Downstream Fine-tuning

#### MNLI

To fine-tune the pre-trained PLMs on MNLI, you need firstly follow `/fairseq-0.9.0/README.glue.md` to download and pre-process dataset. Then 

```
export PYTHONPATH=/fairseq-0.9.0
cd /fairseq-0.9.0
```

To fine-tune pre-trained PLMs with ELLE that takes $\text{BERT}_\text{L6_D384}$ as the initial model:

```
bash eval_MNLI_base_prompt.sh
```

To fine-tune pre-trained PLMs with ELLE that takes $\text{BERT}_\text{L12_D768}$ as the initial model:

```
bash eval_MNLI_large_prompt.sh
```

Note that you need to replace the  **DATA_DIR** and **ROBERTA_PATH** variables in these bash files with your own path to MNLI data folder and model checkpoint path.



#### Others

To fine-tune the pre-trained PLMs on tasks of News, Review, Bio and CS domains, you need firstly follow https://github.com/allenai/dont-stop-pretraining to set up the environment.

You need to organize your PLM checkpoint as `/fairseq_checkpoints/checkpoints_base/checkpoint_last.pt`

and copy the dictionary file:

```
cp /downstream/dict.txt /fairseq_checkpoints/checkpoints_base/
```

Then convert the checkpoint into huggingface checkpoint:

```
cd /downstream
python convert_pnn_to_hf_batch.py /fairseq_checkpoints /hf_checkpoints
cp -r /downstream/base_prompt_files* /hf_checkpoints/checkpoints_base
```

To fine-tune pre-trained PLMs with ELLE that takes $\text{BERT}_\text{L6_D384}$ as the initial model:

```
bash finetune_news.sh
bash finetune_reviews.sh
bash finetune_bio.sh
bash finetune_cs.sh
```

The results are in **roberta_base\_{domain}\_prompt.log**.

To fine-tune pre-trained PLMs with ELLE that takes $\text{BERT}_\text{L12_D768}$ as the initial model, you need to replace the **classifier_prompt_{domain}.jsonnet**  in these bashes with **classifier_1024_prompt_{domain}.jsonnet**.

