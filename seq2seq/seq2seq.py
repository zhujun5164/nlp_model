import numpy as np
import torch
import torch.nn as nn
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
vocab_list = sorted(vocab_dict.items(, key=lambda x: x[1]), reverse=True)
# return list(vocab, appera number)

# 制作idx2str, str2idx
def create_conversion(vocab_list):
    str2idx = {}
    for n, (vocab, _) in enumerate(vocab_list):
        str2idx[vocab] = n
    idx2str = str2idx.keys()
    return idx2str, str2idx

# 对text进行str2idx的转化
sentences_id = []
for sentence in sentences:
    words = sentence.split(' ')
    sentences_id.append([str2idx[word] for word in words])

class model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.enc_LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.dec_LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.enc_fc = nn.Linear(hidden_size * 2, hidden_size * 2)


        self.hidden_size = hidden_size

    def encode(encode_text):
        word_embed = self.embedding(encode_text)
        enc_output, enc_hidden = self.enc_LSTM(word_embed)
        # enc_output: batch_size, seq_size, hidden_size * 2
        # h/c: 2, batch_sizem, hidden_size * 2
        enc_feature = self.enc_fc(enc_output.view(-1, 2), dim = -1)
        # batch_size, seq_size, hidden_size * 2
        return enc_output, enc_hidden, enc_feature

    def decode(dec_len, enc_output, enc_hidden):
        # decode_text: batch_size, seq_len_dec
        # enc_output: batch_size, seq_len_out, hidden_size * 2
        # enc_hidden: 2 * (2, batch_size, hidden_size * 2)
        for word in decode_text:
            # batch_size, 1
            word_embed = self.embedding(decode_word.unsqueeze(1))
            # batch_size, 1, embedding_size
            dec_output, dec_hidden = self.dec_LSTM(word_embed)
            # dec_output: batch_size, 1, hidden_size * 2
            # h/c: 2, batch_size, hidden_size * 2

    def attention(enc_out, )

