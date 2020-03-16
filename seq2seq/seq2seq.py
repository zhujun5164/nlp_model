# import numpy as np
import torch
from collections import Counter

dtype = torch.FloatTensor
sentences = ['ich mocthe ein bier P', 'S i want a beer', 'i want a beer E']

# 首先根据输入文档构建字典
vocab = []
for sentence in sentences:
    words = sentence.split(' ')
    vocab.extend(words)


# 对词频进行统计
vocab_dict = dict(Counter(vocab))

# 对vocab_dict进行排序，然后标号
vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
# return list(vocab, appera number)


# 制作idx2str, str2idx
def create_conversion(vocab_list):
    str2idx = {}
    for n, (vocab, _) in enumerate(vocab_list):
        str2idx[vocab] = n
    idx2str = str2idx.keys()
    return idx2str, str2idx


idx2str, str2idx = create_conversion(vocab_list)

# 对text进行str2idx的转化
sentences_id = []
for sentence in sentences:
    words = sentence.split(' ')
    sentences_id.append([str2idx[word] for word in words])
