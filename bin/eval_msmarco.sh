#!/usr/bin/env bash
# Setting for the new UTF-8 terminal support in Lion
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

run="$1"
qrel="$2"
# echo $fold,$input,$task
python /users/tangyubao/ranker-newk80/bin/ms_marco_eval.py $run $qrel