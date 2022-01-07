#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sample_small_dev.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-11-19 17:34   tangyubao      1.0         None
'''

# import lib
from tqdm import tqdm
path_dev = '/home/tangyubao/msmarco-passage/bm25-top1000.2000'
path_top100_dev = '/home/tangyubao/msmarco-passage/dev.top100.1000'

def save_list(save_file,save_path):
    with open(save_path,'w') as f:
        for s in save_file:
            f.write(s[0]+'\t'+s[1]+'\n')
def load():
    query2doc = {}
    with open(path_dev,'r') as f:
        for line in tqdm(f,desc='read dev'):
            qid, docid, rank = line.split('\t')
            if qid not in query2doc.keys():
                query2doc[qid] = [docid]
            else:
                query2doc[qid].append(docid)
    return query2doc

def process(query2doc):
    output = []
    qid_cnt = 0
    for qid in query2doc.keys():
        if qid_cnt > 999:
            break
        if len(query2doc[qid]) >= 100:
            doc_cnt = 0
            for docid in query2doc[qid] :
                if doc_cnt < 100:
                    output.append([qid,docid])
                    doc_cnt += 1
        qid_cnt += 1
    print('query num:',qid_cnt)
    save_list(output,path_top100_dev)


if __name__ == '__main__':
    query2doc = load()
    process(query2doc)





