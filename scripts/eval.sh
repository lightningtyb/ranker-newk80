#!/bin/bash

python ../decoder/rerank.py \
    --task msmarco \
    --model_name bart_decoder \
    --model_path /data/users/tangyubao/ranker/outputs/models/bart_decoder_ull/epoch-0/ \
    --eval_batch_size 20 \
    --query_file /data/users/tangyubao/msmarco-passage/queries.dev.small.tsv \
    --doc_file /data/users/tangyubao/msmarco-passage/collection_bart_tokenized.txt  \
    --qrel_file /data/users/tangyubao/msmarco-passage/qrels.dev.small.tsv \
    --eval_trec_file /data/users/tangyubao/msmarco-passage/dev.top100.1000 \
    --output_file  /data/users/tangyubao/ranker/outputs/models/bart_decoder_ull/test_eval.run\
    --temp_dir /data/users/tangyubao/ranker/outputs/temp/bart_decoder_ull
