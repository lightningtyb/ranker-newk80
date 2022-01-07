#!/bin/bash

# MS MARCO Document ranking dataset, query num: 372206, valid query num (has pos doc in qrel): 367012
# MS MARCO Passage ranking dataset, query num: , valid query num (has pos doc in qrel): 367012
sleep 1h
python  ../decoder4/trainer.py \
  --task msmarco \
  --model_name bart_decoder \
  --model_path /users/tangyubao/ranker-newk80/bart_basic \
  --warm_up 0.1 \
  --queries_per_epoch 325724 \
  --epochs 10 \
  --patience 300000 \
  --learning_rate 2e-5\
  --negative_num 1 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 2\
  --loss_type mse_neg \
  --strategy q \
  --eval_after_epoch 1 \
  --max_sequence_length 256 \
  --save_model_every_epoch \
  --temp_dir /users/tangyubao/ranker-newk80/outputs/temp/ \
  --query_file /users/tangyubao/msmarco-passage/queries.train_small_dev.tsv \
  --doc_file /users/tangyubao/msmarco-passage/collection_bart_tokenized.txt  \
  --qrel_file /users/tangyubao/msmarco-passage/qrels.train_and_dev.tsv \
  --train_pairs_file /users/tangyubao/msmarco-passage/qidpidtriples.train.small.tsv \
  --eval_trec_file /users/tangyubao/msmarco-passage/dev.top100.1000 \
  --model_out_dir /users/tangyubao/ranker-newk80/outputs/models/decoder_mse_neg_2e_10e/
