TOTAL_UPDATES=62500:20000:20000:20000:20000    # Total number of training steps
WARMUP_RATIO=0.16    # Warmup the learning rate over this many updates
PEAK_LR=0.0005        # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_TOKENS=8192        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
arch=gpt_6layer_384hidden_6head_1536ffn
logdir=log/gpt_base_ELLE
save_dir=checkpoints/gpt_base_ELLE
arch_distil_from=roberta_9_576
models_distil_from=checkpoint_WB_125k.pt:checkpoint_news_125k.pt:checkpoint_reviews_125k.pt:checkpoint_bio_125k.pt:checkpoint_cs_125k.pt
memory_dir=/data/WB_1G:/data/news_1G:/data/reviews_1G:/data/bio_1G:/data/cs_1G
DATA_DIR=/data/WB_17G:/data/news_17G:/data/reviews_17G:/data/bio_17G:/data/cs_17G
DOMAINS=WB:news:reviews:bio:cs
Added_layer_num=2:2:1:1
Added_head_num=2:2:1:1

python ../../fairseq_cli/train.py --fp16 $DATA_DIR \
        --task continual_lm --criterion continual_cross_entropy \
		--save-dir $save_dir \
	    --arch $arch --share-decoder-input-output-embed \
		--dropout 0.1 \
		--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
		--lr $PEAK_LR --lr-scheduler inverse_sqrt --warmup-ratio $WARMUP_RATIO --warmup-init-lr 1e-07 \
		--recover-update 5000\
		--sample-break-mode none --tokens-per-sample $TOKENS_PER_SAMPLE \
		--max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
		--domains $DOMAINS --replay --sample_numbers 200\
		--added_head_num $Added_head_num \
		--added_layer_num $Added_layer_num \
		--restrict_ce_to_mask \
		--arch_distil_from $arch_distil_from \
		--models_distil_from $models_distil_from \
		--replay_ratio 0.1 \
		--validation_interval_updates 1000 \
		--validation_interval_time 1800 \
		--ppl_decreased_min 0.000000001 \
		--add-noise \
		--use_domain_prompt \
		--memory_dir $memory_dir \
		--max-update $TOTAL_UPDATES --log-format json --log-interval 100 \
		--tensorboard-logdir $logdir \
		--skip-invalid-size-inputs-valid-test \
		--fixed-validation-seed 0 \
		--ddp-backend no_c10d \
		--save-interval-updates 5000 \
		--reset-optimizer