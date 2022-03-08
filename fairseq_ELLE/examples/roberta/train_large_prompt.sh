TOTAL_UPDATES=62500:20000:20000:20000:20000    # Total number of training steps
WARMUP_RATIO=0.2    # Warmup the learning rate over this many updates
PEAK_LR=0.00025       # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4        # Number of sequences per batch (batch size)
UPDATE_FREQ=64          # Increase the batch size 16x
arch=roberta_12layer_768hidden_12head_3072ffn
logdir=log/roberta_large_ELLE
save_dir=checkpoints/roberta_large_ELLE
arch_distil_from=roberta_9_576
models_distil_from=checkpoint_WB_125k.pt:checkpoint_news_125k.pt:checkpoint_reviews_125k.pt:checkpoint_bio_125k.pt:checkpoint_cs_125k.pt
memory_dir=/data/WB_1G:/data/news_1G:/data/reviews_1G:/data/bio_1G:/data/cs_1G
DATA_DIR=/data/WB_17G:/data/news_17G:/data/reviews_17G:/data/bio_17G:/data/cs_17G
DOMAINS=WB:news:reviews:bio:cs
Added_layer_num=3:3:3:3
Added_head_num=3:1:0:0

python ../../fairseq_cli/train.py --fp16 $DATA_DIR \
        --task continual_KI --criterion continual_KI \
	    --arch $arch --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
		--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
		--lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-ratio $WARMUP_RATIO --total-num-update $TOTAL_UPDATES --recover-update 1000\
		--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
		--domains $DOMAINS --replay --sample_numbers 200\
		--added_head_num $Added_head_num \
		--added_layer_num $Added_layer_num \
		--restrict_ce_to_mask \
		--arch_distil_from $arch_distil_from \
		--models_distil_from $models_distil_from \
		--replay_ratio 0.1 \
		--validation_interval_updates 1000 \
		--ppl_decreased_min 0.000000001 \
        --use_domain_prompt \
		--memory_dir $memory_dir \
		--max-update $TOTAL_UPDATES --log-format json --log-interval 100 \
		--tensorboard-logdir $logdir \
		--skip-invalid-size-inputs-valid-test \
		--save-dir $save_dir \
		--fixed-validation-seed 0 \
		--ddp-backend no_c10d \
		--save-interval-updates 5000 \
		--reset-optimizer