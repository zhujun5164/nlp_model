# import numpy as np
import torch
import numpy as np
from collections import Counter

dtype = torch.FloatTensor
# 假设吧= =，假设
enc_sentences = ['ich mocthe ein bier P', 'S i want a beer', 'i want a beer E']
 = ['sdn qoi sdq', 'dq wdsf', 'sdqn qw ceq']

for i in len(enc_sentences):
    enc_sentences[i] = 'BOS ' + enc_sentences[i] + ' EOS'
    dec_sentences[i] + 'BOS ' + dec_sentences[i] + ' EOS'

# 首先根据输入文档构建字典
vocab = []
for sentence in enc_sentences:
    words = sentence.split(' ')
    vocab.extend(words)
for sentence in dec_sentences:
    words = sentence.split(' ')
    vocab.extend(words)

# 对词频进行统计
vocab_dict = dict(Counter(vocab))

# 对vocab_dict进行排序，然后标号
vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
# return list(vocab, appera number)


# 制作idx2str, str2idx
def create_conversion(vocab_list):
    str2idx = {'PAD':0, 'UNK':1}
    for n, (vocab, _) in enumerate(vocab_list):
        str2idx[vocab] = n + 4
    idx2str = str2idx.keys()
    return idx2str, str2idx


idx2str, str2idx = create_conversion(vocab_list)

# 对text进行str2idx的转化
sentences_id = []
for sentence in sentences:
    words = sentence.split(' ')
    sentences_id.append([str2idx[word] if word in idx2str else str2idx['UNK'] for word in words])

# 获取训练数据
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def get_examples(enc_sentences, dec_sentences, batch_size):
    minibatches = get_minibatches(len(enc_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        