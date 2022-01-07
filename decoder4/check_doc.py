#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   check_doc.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-11-20 17:48   tangyubao      1.0         None
'''

# import lib
from tqdm import tqdm
path_doc = '/home/tangyubao/msmarco-passage/collection.tsv'
path_trec = '/home/tangyubao/msmarco-passage/dev.100'

def load_doc():
    doc_ids = []
    with open(path_doc,'r') as f:
        for line in tqdm(f,desc='loading doc'):
            cols = line.rstrip().split('\t')
            doc_ids.append(cols[0])
    return doc_ids

def load_trec(doc_ids):
    cnt = 0
    line_cnt = 0
    with open(path_trec,'r') as f:
        for line in tqdm(f,desc='loading trec'):
            qid,did = line.split('\t')
            if qid not in doc_ids:
                cnt += 1
            line_cnt += 1
    print('not in doc num:',cnt,'total qrel num:',line_cnt)

if __name__ == '__main__':
    doc_ids=load_doc()
    load_trec(doc_ids)

