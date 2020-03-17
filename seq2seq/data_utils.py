import unicodedata
import re
from collections import Counter
import numpy as np


# -----------------------------------------------------------------------------
# 对unicode字符转为acsii码
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # 因为文本中标点符号是连着单词的，因此通过正则识别出来进行加“空格”
    s = re.sub(r"([.!?])", r" \1", s)
    # 把不是英文单词的内容都变为“空格”
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# http://www.manythings.org/anki/
def read_anki_data(data_path):
    datas1, datas2 = [], []
    with open(data_path, 'r', encoding='utf-8') as tf:
        for i in tf.readlines():
            data1, data2, _ = i.lower().strip().split('\t')
            datas1.append(normalizeString(data1))
            datas2.append(normalizeString(data2))
    return datas1, datas2


# ---------------------------------------------------------------------------
def data_prepared(data_sentences):
    vocab = []
    for i in range(len(data_sentences)):
        data_sentences[i] = 'BOS ' + data_sentences[i] + ' EOS'
        vocab.extend(data_sentences[i].split(' '))

    # Count vocab number
    vocab_dict = dict(Counter(vocab))
    # sort by appear count
    vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)

    word2id = {'PAD': 0, 'UNK': 1}
    id2word = {0: 'PAD', 1: 'UNK'}
    for n, (vocab, _) in enumerate(vocab_list):
        word2id[vocab] = n + 2
        id2word[n+2] = vocab
    return data_sentences, word2id, id2word, len(word2id)


# ------------------------------------------------------------------------------------
# 获取batch数据，并padding
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
        # 每个batch使用数据的下标号
    return minibatches


def pad_data(data):
    seq_len = [len(sentence) for sentence in data]
    max_len = max(seq_len)
    n_samples = len(data)

    # 我们之前定义了pad是代表0，所以才使用np.zeros。若pad不为0，可以后面 + str2idx['PAD']
    pad_data = np.zeros((n_samples, max_len))
    mask_metrix = np.ones((n_samples, max_len))
    for i in range(n_samples):
        pad_data[i, :seq_len[i]] = data[i]

        # 若被mask，则为1；如果为正文，则为0
        mask_metrix[i, seq_len[i]:] = 0
    return pad_data, mask_metrix


def get_examples(enc_sentences, dec_sentences, batch_size):
    minibatches = get_minibatches(len(enc_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        enc_batch, dec_batch = [], []
        enc_batch = [enc_sentences[num] for num in minibatch]
        dec_batch = [dec_sentences[num] for num in minibatch]
        enc_data, enc_mask = pad_data(enc_batch)
        dec_data, dec_mask = pad_data(dec_batch)
        all_ex.append((enc_data, enc_mask, dec_data, dec_mask))
    return all_ex


# -----------------word2id, id2word--------------------
def word2id(sentences, word2id_dict):
    # sentences [[sdq dqw dw vr], [wq fge qw sv], ...]
    word_ids = []
    for i in range(len(sentences)):
        words = sentences[i].split(' ')
        word_ids.append(word2id_dict[word] if word in word2id_dict.keys() else word2id_dict['UNK'] for word in words)
    return word_ids


def id2word(sentences, id2word_dict):
    # sentences [[1, 2, 3, 4, ..], [2, 4, 5, 7, ..], ...]
    sentences = []
    for i in range(len(sentences)):
        words = []
        for id in sentences[i]:
            word = id2word_dict[id]
            if word != 'EOS':
                words.append(word)
            else:
                break
        sentences.append(words)
    return sentences
