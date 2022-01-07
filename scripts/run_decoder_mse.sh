#!/bin/bash

# MS MARCO Document ranking dataset, query num: 372206, valid query num (has pos doc in qrel): 367012
# MS MARCO Passage ranking dataset, query num: , valid query num (has pos doc in qrel): 367012

python  ../decoder/train.py \
  --task msmarco \
  --model_name bart_decoder \
  --model_path facebook/bart-large \
  --warm_up 0.1 \
  --queries_per_epoch 200000 \
  --epochs 10 \
  --patience 300000 \
  --learning_rate 1e-5\
  --negative_num 1 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps  32\
  --loss_type mse \
  --strategy q \
  --eval_after_epoch 1 \
  --max_sequence_length 256 \
  --save_model_every_epoch \
  --temp_dir /home/tangyubao/bert-reranker-main/outputs/temp/mse_test/ \
  --query_file /home/tangyubao/msmarco-passage/queries.train_small_dev.tsv \
  --doc_file /home/tangyubao/msmarco-passage/collection_bart_tokenized.txt  \
  --qrel_file /home/tangyubao/msmarco-passage/qrels.train_and_dev.tsv \
  --train_pairs_file /home/tangyubao/msmarco-passage/qidpidtriples.train.small.tsv \
  --eval_trec_file /home/tangyubao/msmarco-passage/dev.top100.1000 \
  --model_out_dir /home/tangyubao/bert-reranker-main/outputs/models/mse_test/
