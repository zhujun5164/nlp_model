import torch
import torch.nn as nn


class seq2seq_atten(nn.Module):
    def __init__(self, vocab_size_enc, vocab_size_dec, embedding_size, hidden_size):
        super(seq2seq_atten, self).__init__()
        self.enc_embedding = nn.Embedding(vocab_size_enc, embedding_size)
        self.enc_LSTM = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)

        self.dec_embedding = nn.Embedding(vocab_size_dec, embedding_size)
        self.dec_LSTM = nn.LSTM(embedding_size + hidden_size, hidden_size, batch_first=True, bidirectional=False)

        self.enc_h_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_c_fc = nn.Linear(hidden_size * 2, hidden_size)

        self.dec_feature_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.enc_output_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.all_feature_fc = nn.Linear(hidden_size * 2, 1)

        self.dec_out = nn.Linear(hidden_size * 2, vocab_size_dec)

        self.hidden_size = hidden_size

    def encode(self, encode_text, encode_mask):
        lengths = torch.sum(encode_mask, dim=1)

        embeded = self.embedding(encode_text)  # batch_size, seq_len, embedding_size
        embeded = torch.nn.utils.rnn.pack_padded_sequence(input=embeded, lengths=lengths, batch_first=True, enforce_sorted=False)
        enc_output, enc_hidden = self.enc_LSTM(embeded)
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=enc_output, batch_first=True, padding_value=0.0, total_length=torch.max(lengths))
        # enc_output: batch_size, seq_size, hidden_size * 2
        # h/c: 2, batch_size, hidden_size

        return enc_output, enc_hidden

    def decode_init(self, enc_hidden):
        # enc_hidden: 2 * (2, batch_size, hidden_size)
        batch_size = enc_hidden[0].shape[1]

        # init_dec_input
        # dec_input = torch.zeros((batch_size, 1))  # batch_size, 1 -- 初始的输入， 如果对文本最前面加上了BOS的话，也可以根据text来进行一个个取
        dec_context = torch.zeros((batch_size, 1, self.hidden_size))  # batch_size, 1, hidden_size -- 包含前文的特征，最初认为是什么都看不了，所以是0

        # transfor enc_hidden
        # 对enc_hidden进行转化，得到适用于dec层输入的hidden_states
        # enc_hidden -> (h_n, c_n) -> 2 * (2, batch_size, hidden_size)
        # dec_hidden -> (h_n, c_n) -> 2 * (1, batch_size, hidden_size)
        # 对enc_hidden 进行处理， 2， batch_size, hidden_size -> batch_size, hidden_size * 2
        #                            batch_size, hidden_size * 2 -relu(linear)-> batch_size, hidden_size
        # 动起来~
        enc_h, enc_c = enc_hidden  # (2, batch_size, hidden_size)
        enc_h = enc_h.permute(2, 0, 1).view(-1, self.hidden_size * 2)
        dec_h = torch.relu(self.enc_h_fc(enc_h)).unsqueeze(0)  # 1, batch_size, hidden_size

        enc_c = enc_c.permute(2, 0, 1).view(-1, self.hidden_size * 2)
        dec_c = torch.relu(self.enc_c_fc(enc_c)).unsqueeze(0)  # 1, batch_size, hidden_size

        dec_hidden = (dec_h, dec_c)  # 2 * (1, batch_size, hidden_size)

        # return dec_input, dec_context, dec_hidden
        return dec_context, dec_hidden

    def atten(self, enc_output, enc_mask, dec_hidden, attention='dot'):
        # enc_output [batch_size, seq_len, hidden_size * 2]
        # dec_hidden 2 * ([1, batch_size, hidden_size])

        # 对dec_output进行处理， 得到batch_size, 1, hidden_size * 2的数据
        dec_feature = torch.cat([dec_hidden[0], dec_hidden[1]], dim=2)
        # 1, batch_size, hidden_size * 2

        dec_feature = self.dec_feature_fc(dec_feature).unsqueeze(1)
        enc_feature = self.enc_output_fc(enc_output, dim=2)
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
        atten = (atten / atten.sum(dim=1, keepdim=True)).unsqueeze(0)

        dec_context = torch.bmm(atten, enc_output)
        # batch_size, 1, hidden_size
        return atten, dec_context

    # train
    def decode(self, enc_output, enc_mask, dec_input, dec_context, dec_hidden):

        # 在train步骤上，输入的是dec文档的文本内容,若为预测步骤则需要将上一个输出的预测结果用作输入，最开始输入的应该是启动标识符
        dec_input = self.embedding(dec_input)
        # batch_size, 1, embedding_size

        # 这一操作其实就有点像BahdanauAttn
        dec_input = torch.cat((dec_input, dec_context), dim=-1)
        # batch_size, 1, embedding_size + hidden_size

        dec_output, dec_hidden = self.dec_LSTM(dec_input, dec_hidden)
        # dec_output: batch_size, 1, hidden_size
        # h/c: 1, batch_size, hidden_size

        # 这一个attention用作输出的是 Luong attention
        atten, dec_context = self.atten(enc_output, enc_mask, dec_hidden)
        # batch_size, 1, seq_len_enc
        # batch_size, 1, hidden_size

        dec_output = self.dec_out(torch.cat((dec_output, dec_context), dim=2), dim=2).squeeze(1)
        # batch_size, num_classes
        return dec_output, dec_context, dec_hidden, atten

    def forward(self, enc_texts, enc_masks, dec_texts, dec_masks, loss_fn):
        # enc_text: batch_size, seq_len
        # target: batch_size, seq_len
        enc_output, enc_hidden = self.encode(enc_texts)
        # dec_input, dec_context, dec_hidden = self.decode_init(enc_hidden)
        dec_context, dec_hidden = self.decode_init(enc_hidden)

        dec_batch_size, dec_seq_len = dec_texts.shape

        loss = 0
        predict = []

        use_teacher_forcing = torch.random() > 0.5
        if use_teacher_forcing:
            for i in range(dec_seq_len - 1):
                dec_input = dec_texts[:, i]
                dec_target = dec_texts[:, i + 1]

                dec_output, dec_context, dec_hidden, atten = self.decode(enc_output, enc_masks, dec_input, dec_context, dec_hidden)

                # compute loss
                loss_step = loss_fn(dec_output, dec_target)  # batch_size
                loss_step = loss_step.mul(dec_masks[:, i + 1])
                loss += loss_step / torch.sum(dec_masks[:, i + 1])

                _, step_predict = (torch.softmax(dec_output).data).topk(1)  # batch_size, 1
                predict.append(step_predict)

        else:
            for i in range(dec_seq_len - 1):

                if i == 0:
                    dec_input = dec_texts[:, i]
                dec_target = dec_texts[:, i + 1]

                dec_output, dec_context, dec_hidden, atten = self.decode(enc_output, enc_masks, dec_input, dec_context, dec_hidden)
                _, dec_input = (torch.softmax(dec_output).data).topk(1)  # batch_size, 1

                # compute loss
                loss_step = loss_fn(dec_output, dec_target)  # batch_size
                loss_step = loss_step.mul(dec_masks[:, i + 1])
                loss += loss_step / torch.sum(dec_masks[:, i + 1])

                predict.append(dec_input)

        predict = torch.stack((predict[:]), dim=-1)

        return loss, predict


class easy_seq2seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes):
        super(easy_seq2seq, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.enc_LSTM = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.h_to_dec_h = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.c_to_dec_c = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.dec_LSTM = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.hidden_size = hidden_size

    def encode(self, text):
        # [batch_size, seq_len]
        embeded = self.embedding(text)  # [batch_size, seq_len ,embedding_size]
        _, enc_hidden = self.enc_LSTM(embeded)
        # [batch_size, seq_len, hidden_size * 2], 2 * [2, batch_size, hidden_size]
        return enc_hidden

    def decode(self, text, dec_hidden):
        # [batch_size, seq_len]
        embeded = self.embedding(text)
        embeded = torch.relu(embeded)
        output, dec_hidden = self.dec_LSTM(embeded, dec_hidden)
        return output

    def feature_translater(self, enc_hidden):
        # 2 * [2, batch_size, hidden_size] -> 2 * [1, batch_size, hidden_size]
        enc_h, enc_c = enc_hidden
        enc_h = enc_h.permute(2, 0, 1).view(-1, self.hidden_size * 2)
        dec_init_h = torch.relu(self.h_to_dec_h(enc_h))

        enc_c = enc_c.permute(2, 0, 1).view(-1, self.hidden_size * 2)
        dec_init_c = torch.relu(self.c_to_dec_c(enc_c))
        hidden = (dec_init_h, dec_init_c)
        return hidden

    def forward(self, encode_text, target_text):
        enc_hidden = self.encode(encode_text)
        dec_hidden = self.feature_translater(enc_hidden)
        output = self.decode(target_text, dec_hidden)

        output = torch.softmax(self.classifier(output), dim=-1)
        return output
