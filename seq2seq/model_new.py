import torch
import torch.nn as nn
import random
from tqdm import tqdm

rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(rand_unif_init_mag, rand_unif_init_mag)


class seq2seq_atten(nn.Module):
    def __init__(self, vocab_size_enc, vocab_size_dec, embedding_size, hidden_size):
        super(seq2seq_atten, self).__init__()
        self.enc_embedding = nn.Embedding(vocab_size_enc, embedding_size)
        init_wt_normal(self.enc_embedding)
        self.enc_LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        init_lstm_wt(self.enc_LSTM)

        self.dec_embedding = nn.Embedding(vocab_size_dec, embedding_size)
        init_wt_normal(self.dec_embedding)
        self.dec_input_fc = nn.Linear(embedding_size + hidden_size * 2, embedding_size)
        init_linear_wt(self.dec_input_fc)
        self.dec_LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=False)
        init_linear_wt(self.dec_LSTM)

        self.enc_h_fc = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.enc_h_fc)
        self.enc_c_fc = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.enc_c_fc)

        self.dec_feature_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        init_linear_wt(self.dec_feature_fc)
        self.enc_output_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        init_linear_wt(self.enc_output_fc)
        self.all_feature_fc = nn.Linear(hidden_size * 2, 1)
        init_linear_wt(self.all_feature_fc)

        self.dec_out = nn.Linear(hidden_size * 2 + hidden_size, vocab_size_dec)
        init_linear_wt(self.dec_out)

        self.hidden_size = hidden_size
        self.vocab_size_dec = vocab_size_dec

    def encode(self, encode_text, encode_mask):
        lengths = torch.sum(encode_mask, dim=1)

        embeded = self.enc_embedding(encode_text)  # batch_size, seq_len, embedding_size
        embeded = torch.nn.utils.rnn.pack_padded_sequence(input=embeded, lengths=lengths, batch_first=True, enforce_sorted=False)
        enc_output, enc_hidden = self.enc_LSTM(embeded)
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=enc_output, batch_first=True, padding_value=0.0, total_length=int(torch.max(lengths).item()))
        # enc_output: batch_size, seq_size, hidden_size * 2
        # h/c: 2, batch_size, hidden_size

        return enc_output, enc_hidden

    def decode_init(self, enc_hidden, BOS_num):
        # enc_hidden: 2 * (2, batch_size, hidden_size)
        batch_size = enc_hidden[0].shape[1]

        # init_dec_input
        dec_input = torch.zeros((batch_size, 1)).fill_(BOS_num)  # batch_size, 1 -- 初始的输入， 如果对文本最前面加上了BOS的话，也可以根据text来进行一个个取
        dec_context = torch.zeros((batch_size, 1, self.hidden_size * 2))  # batch_size, 1, hidden_size * 2-- 包含前文的特征，最初认为是什么都看不了，所以是0

        # transfor enc_hidden
        # 对enc_hidden进行转化，得到适用于dec层输入的hidden_states
        # enc_hidden -> (h_n, c_n) -> 2 * (2, batch_size, hidden_size)
        # dec_hidden -> (h_n, c_n) -> 2 * (1, batch_size, hidden_size)
        # 对enc_hidden 进行处理， 2， batch_size, hidden_size -> batch_size, hidden_size * 2
        #                            batch_size, hidden_size * 2 -relu(linear)-> batch_size, hidden_size
        # 动起来~
        enc_h, enc_c = enc_hidden  # (2, batch_size, hidden_size)
        enc_h = enc_h.permute(2, 0, 1).reshape(-1, self.hidden_size * 2)
        dec_h = torch.relu(self.enc_h_fc(enc_h)).unsqueeze(0)  # 1, batch_size, hidden_size

        enc_c = enc_c.permute(2, 0, 1).reshape(-1, self.hidden_size * 2)
        dec_c = torch.relu(self.enc_c_fc(enc_c)).unsqueeze(0)  # 1, batch_size, hidden_size

        dec_hidden = (dec_h, dec_c)  # 2 * (1, batch_size, hidden_size)

        # return dec_input, dec_context, dec_hidden
        return dec_input, dec_context, dec_hidden

    def atten(self, enc_output, enc_mask, dec_hidden, attention='dot'):
        # enc_output [batch_size, seq_len, hidden_size * 2]
        # dec_hidden 2 * ([1, batch_size, hidden_size])

        # 对dec_output进行处理， 得到batch_size, 1, hidden_size * 2的数据
        dec_feature = torch.cat([dec_hidden[0], dec_hidden[1]], dim=2)
        # 1, batch_size, hidden_size * 2

        dec_feature = self.dec_feature_fc(dec_feature).permute(1, 0, 2)
        enc_feature = self.enc_output_fc(enc_output)
        # batch_size, 1, hidden_size * 2

        # 计算输出与enc_output的权重
        # [batch_size, 1, hidden_size * 2]  [batch_size, seq_size, hidden_size * 2]
        if attention == 'dot':
            scores = torch.bmm(dec_feature, enc_feature.permute(0, 2, 1))
        else:  # concat
            all_feature = torch.tanh(dec_feature + enc_feature)
            scores = self.all_feature_fc(all_feature, dim=2)

        atten = torch.softmax(scores, dim=-1)
        # batch_size, 1, seq_len  batch_size, seq_len
        atten = torch.mul(atten.squeeze(), enc_mask)
        atten = (atten / atten.sum(dim=1, keepdim=True)).unsqueeze(1)

        dec_context = torch.bmm(atten, enc_output)
        # batch_size, 1, hidden_size
        return atten, dec_context

    # train
    def decode(self, enc_output, enc_mask, dec_input, dec_context, dec_hidden):

        # 在train步骤上，输入的是dec文档的文本内容,若为预测步骤则需要将上一个输出的预测结果用作输入，最开始输入的应该是启动标识符
        dec_input = self.dec_embedding(dec_input).unsqueeze(1)
        # batch_size, 1, embedding_size

        # 这一操作其实就有点像BahdanauAttn
        dec_input = torch.cat((dec_input, dec_context), dim=-1)
        # batch_size, 1, embedding_size + hidden_size * 2
        dec_input = self.dec_input_fc(dec_input)

        dec_output, dec_hidden = self.dec_LSTM(dec_input, dec_hidden)
        # dec_output: batch_size, 1, hidden_size
        # h/c: 1, batch_size, hidden_size

        # 这一个attention用作输出的是 Luong attention
        atten, dec_context = self.atten(enc_output, enc_mask, dec_hidden)
        # batch_size, 1, seq_len_enc
        # batch_size, 1, hidden_size

        dec_output = self.dec_out(torch.cat((dec_output, dec_context), dim=2)).squeeze(1)
        # batch_size, num_classes
        return dec_output, dec_context, dec_hidden, atten

    def loss(self, loss_fn, predict, target, mask):
        # predict [batch_size,  N]
        # target [batch_size]

        # 之前做错做了个BCEWithlogitloss的，保存下
        # one_hot_target = torch.zeros((dec_target.shape[0], self.vocab_size_dec)).scatter_(1, dec_target.unsqueeze(1), 1)
        #     loss_step = loss_fn(dec_output, one_hot_target)  # batch_size, dec_vocab_size
        #     loss_step = loss_step.mul((dec_masks[:, i + 1]).unsqueeze(1))  # batch_size, dec_vocab_size

        loss = loss_fn(predict, target)
        loss = loss.mul(mask)

        return loss

    def forward(self, enc_texts, enc_masks, dec_texts=None, dec_masks=None, loss_fn=None):
        # enc_text: batch_size, seq_len
        # target: batch_size, seq_len
        enc_output, enc_hidden = self.encode(enc_texts, enc_masks)
        # dec_input, dec_context, dec_hidden = self.decode_init(enc_hidden)
        dec_context, dec_hidden = self.decode_init(enc_hidden)

        dec_batch_size, dec_seq_len = dec_texts.shape

        loss = []
        predict = []

        use_teacher_forcing = True if random.random() > 0. else False
        if use_teacher_forcing:
            for i in range(dec_seq_len - 1):
                dec_input = dec_texts[:, i]
                dec_target = dec_texts[:, i + 1]

                dec_output, dec_context, dec_hidden, atten = self.decode(enc_output, enc_masks, dec_input, dec_context, dec_hidden)

                # compute loss
                loss_step = self.loss(loss_fn, dec_output, dec_target, dec_masks[:, i + 1])
                # 希望输出的是batch_size

                loss.append(loss_step)

                # get_predict
                _, step_predict = (torch.softmax(dec_output, dim=-1).data).topk(1)  # batch_size, 1
                predict.append(step_predict.squeeze())

        else:
            for i in range(dec_seq_len - 1):
                if i == 0:
                    dec_input = dec_texts[:, i]
                dec_target = dec_texts[:, i + 1]

                dec_output, dec_context, dec_hidden, atten = self.decode(enc_output, enc_masks, dec_input, dec_context, dec_hidden)
                _, dec_input = (torch.softmax(dec_output, dim=-1).data).topk(1)  # batch_size, 1
                dec_input = dec_input.squeeze()

                # compute loss
                loss_step = self.loss(loss_fn, dec_output, dec_target, dec_masks[:, i + 1])
                # 希望输出的是batch_size

                loss.append(loss_step)
                predict.append(dec_input)

        predict = torch.stack((predict[:]), dim=-1)

        loss = torch.stack((loss[:]), dim=-1)
        loss = torch.mean(loss.sum(dim=1) / dec_masks.sum(dim=1))

        return loss, predict


def train_one_batch(model, data):
    model = seq2seq_atten(1, 1, 1, 1)
    loss_fn = 1
    optim = 1
    data = 1

    for enc_texts, enc_masks, dec_texts, dec_masks in tqdm(data, ncols=80):

        model.train()

        enc_data = torch.LongTensor(enc_texts)  # batch_size, seq_len
        enc_mask = torch.as_tensor(enc_masks).float()  # batch_size, seq_len

        dec_data = torch.LongTensor(dec_texts)  # batch_size, seq_len
        dec_mask = torch.as_tensor(dec_masks).float()  # batch_size, seq_len

        optim.zero_grad()

        # train

        # encode
        enc_output, enc_hidden = model.encode(enc_data, enc_mask)
        # init_decode   需要改，还是得生成一个BOS的input
        dec_input, dec_context, dec_hidden = model.decode_init(enc_hidden, BOS_num=1)

        # decode
        loss, predict = [], []
        dec_seq_len = dec_data.shape[1]

        use_teacher_forcing = True if random.random() > 0.5 else False

        for i in range(dec_seq_len - 1):
            dec_target = dec_texts[:, i + 1]
            dec_mask = dec_masks[:, i + 1]

            dec_output, dec_context, dec_hidden, atten = model.decode(enc_output, enc_mask, dec_input, dec_context, dec_hidden)

            loss_step = model.loss(loss_fn, dec_output, dec_target, dec_mask)
            loss.append(loss_step)

            _, predict_step = (torch.softmax(dec_output, dim=-1).data).topk(1)  # batch_size, 1
            predict_step = predict_step.squeeze()
            predict.append(predict_step)

            if use_teacher_forcing:
                dec_input = dec_texts[:, i]

            else:
                dec_input = predict_step

        loss = torch.stack((loss[:]), dim=-1)
        loss = torch.mean(loss.sum(dim=1) / dec_masks.sum(dim=1))

        predict = torch.stack((predict[:]), dim=-1)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
        optim.step()

    return loss


def eval(model, data):
    model = seq2seq_atten(1, 1, 1, 1)

    data = 1
    max_len = 1

    for enc_texts, enc_masks in tqdm(data, ncols=80):

        model.eval()

        enc_data = torch.LongTensor(enc_texts)  # batch_size, seq_len
        enc_mask = torch.as_tensor(enc_masks).float()  # batch_size, seq_len

        # eval
        with torch.no_grad():
            # encode
            enc_output, enc_hidden = model.encode(enc_data, enc_mask)
            # init_decode   需要改，还是得生成一个BOS的input
            dec_input, dec_context, dec_hidden = model.decode_init(enc_hidden, BOS_num=1)

            # decode
            predict = [], []

            for i in range(max_len):

                dec_output, dec_context, dec_hidden, atten = model.decode(enc_output, enc_mask, dec_input, dec_context, dec_hidden)

                _, predict_step = (torch.softmax(dec_output, dim=-1).data).topk(1)  # batch_size, 1
                predict_step = predict_step.squeeze()
                predict.append(predict_step)

                dec_input = predict_step

            predict = torch.stack((predict[:]), dim=-1)

    return predict
