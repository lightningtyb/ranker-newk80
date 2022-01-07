#!/bin/bash

# MS MARCO Document ranking dataset, query num: 372206, valid query num (has pos doc in qrel): 367012
# MS MARCO Passage ranking dataset, query num: , valid query num (has pos doc in qrel): 367012

python  ../src/trainer.py \
  --task msmarco \
  --model_name bert \
  --model_path bert-base-uncased \
  --warm_up 0.1 \
  --queries_per_epoch 367012 \
  --epochs 10 \
  --patience 3000000 \
  --learning_rate 1e-5\
  --negative_num 5 \
  --train_batch_size 24 \
  --eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --loss_type ce \
  --strategy q \
  --eval_after_epoch 1 \
  --max_sequence_length 512 \
  --save_model_every_epoch \
  --temp_dir ./ \
  --query_file /home/tangyubao/msmarco-ma/queries-original \
  --doc_file /home/tangyubao/msmarco-ma/collection_bert_tokenized.txt  \
  --qrel_file /home/tangyubao/msmarco-ma/qrels \
  --train_pairs_file /home/tangyubao/msmarco-ma/train_pairs_top100 \
  --eval_trec_file /home/tangyubao/msmarco-ma/small_dev_run_1000 \
  --model_out_dir tmp/
