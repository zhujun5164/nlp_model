import torch
from data_utils import read_anki_data, data_prepared, word2id, get_examples
from model import seq2seq_atten

# config
data_path = 'D:/data/nlp_model/seq2seq/fra.txt'
batch_size = 8
embedding_size = 256
hidden_size = 256
learning_rate = 1e-4
epoches = 1

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# 读取数据
enc_sentences, dec_sentences = read_anki_data(data_path)

enc_sentences, word2id_enc, id2word_enc, vocab_size_enc = data_prepared(enc_sentences)
dec_sentences, word2id_dec, id2word_dec, vocab_size_dec = data_prepared(dec_sentences)

enc_id = word2id(enc_sentences, word2id_enc)
dec_id = word2id(dec_sentences, word2id_dec)

# get_data
data = get_examples(enc_id, dec_id, batch_size)

# create model
model = seq2seq_atten(vocab_size_enc, vocab_size_dec, embedding_size, hidden_size)
model = model.to(device)

# get training parameters
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad is True:
        params_to_update.append(param)

optim = torch.optim.Adam(params_to_update, lr=learning_rate)

loss_fn = torch.nn.BCEWithLogitsLoss()

for n in range(epoches):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    n = 0
    for enc_data, enc_mask, dec_data, dec_mask in data:
        # batch_size, seq_len
        enc_data = torch.as_tensor(enc_data).float()  # batch_size, seq_len
        enc_mask = torch.as_tensor(enc_mask)  # batch_size, seq_len

        dec_data = torch.as_tensor(dec_data).float()  # batch_size, seq_len
        dec_mask = torch.as_tensor(dec_mask)  # batch_size, seq_len

        predict = model(enc_data, enc_mask, dec_data, dec_mask)
