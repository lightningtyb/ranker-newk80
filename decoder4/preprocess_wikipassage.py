#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess_wikipassage.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-12-07 10:14   tangyubao      1.0         None
'''

# import lib
from tqdm import tqdm
import json
path_query = '/home/tangyubao/wikipassage/train.tsv'
path_out_query = '/home/tangyubao/wikipassage/queries.train.tsv'
path_out_qrel = '/home/tangyubao/wikipassage/qrel.train.tsv'

path_doc = '/home/tangyubao/wikipassage/document_passages.json'


def save_list(save_file,save_path):
    with open(save_path,'w') as f:
        for s in save_file:
            f.write(s+'\n')


# query = []
# qrel = []
# with open(path_query, 'r') as f_d:
#     for line in tqdm(f_d, desc='processing query'):
#         l = line.split('\t')
#         qid, q, did, dname, rep = l
#         if l[2] not in title:
#             title.append(l[2])

with open(path_doc,'r') as doc:
    content = json.load(doc)
    for k, v in content.items():
        print(k,v)
        break