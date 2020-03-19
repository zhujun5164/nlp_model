import torch
from data_utils import read_anki_data, data_prepared, word2id, get_examples
from model import seq2seq_atten
from tqdm import tqdm
# from tensorboard import summary


def train(model, enc_data, enc_mask, dec_data, dec_mask):

    model.train()

    enc_data = torch.LongTensor(enc_data)  # batch_size, seq_len
    enc_mask = torch.as_tensor(enc_mask).float()  # batch_size, seq_len

    dec_data = torch.LongTensor(dec_data)  # batch_size, seq_len
    dec_mask = torch.as_tensor(dec_mask).float()  # batch_size, seq_len

    optim.zero_grad()
    loss, predict = model(enc_data, enc_mask, dec_data, dec_mask, loss_fn)

    loss.backward()

    torch.nn.utils.clip_grad_norm(params_to_update, 5.0)
    optim.step()

    return loss, predict


def eval(model, enc_data, enc_mask):

    model.eval()

    enc_data = torch.LongTensor(enc_data)  # batch_size, seq_len
    enc_mask = torch.as_tensor(enc_mask).float()  # batch_size, seq_len

    predict = model(enc_data, enc_mask)
    return predict


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

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

i = 0
for n in range(epoches):

    running_loss = 0.0
    running_corrects = 0

    for enc_data, enc_mask, dec_data, dec_mask in tqdm(data, ncols=80):
        i += 1
        loss, predict = train(model, enc_data, enc_mask, dec_data, dec_mask)

    # eval
