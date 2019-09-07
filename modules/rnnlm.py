'''
RNN Language Model：
基于RNN的语言模型，上下文不再是n-gram而是整个上文句子
'''
import torch.nn as nn
import torch
import numpy as np
from .rnn_encoder import RNNEncoder

'''
方式一：以截取的定长序列作为输入，序列的下一个词为预测输出
总结：
    1、截取的长度对结果影响不大
    2、训练时batch_size尽量设的大一点

'''


class RNNLanguageModel(nn.Module):
    def __init__(self, args):
        super(RNNLanguageModel, self).__init__()

        self.args = args

        self.hidden_size = args.hidden_size

        # 副产品
        self.embed_layer = nn.Embedding(num_embeddings=args.vocab_size,
                                        embedding_dim=args.word_size,
                                        padding_idx=0)

        nn.init.xavier_uniform_(self.embed_layer.weight)

        # 不定长序列需要pack-pad sequence
        self.rnn_layer = nn.LSTM(
            input_size=args.word_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.drop_rnn,
            batch_first=True,
            bidirectional=False
        )

        # self.rnn_layer = RNNEncoder(
        #     input_size=args.word_size,
        #     hidden_size=args.hidden_size,
        #     num_layers=args.num_layers,
        #     dropout=args.drop_rnn,
        #     batch_first=True,
        #     bidirectional=False
        # )

        self.dropout_embed = nn.Dropout(args.drop_embed)
        self.dropout_layer = nn.Dropout(args.drop_rnn)

        self.linear = nn.Linear(in_features=args.hidden_size,
                                out_features=args.vocab_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        h0 = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        # h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0, h0

    def save_embed_weights(self, path):
        np.save(path, self.embed_layer.weight.data.numpy())

    def forward(self, inputs, hidden=None):
        '''
        :param inputs: (batch_size, seq_len)
        :param hidden: (batch_size, hidden_size)
        :return:
        '''
        batch_size = inputs.size(0)

        # (batch_size, seq_len, embed_dim)
        embed = self.embed_layer(inputs)

        if self.training:
            embed = self.dropout_embed(embed)

        if hidden is None:
            hidden = self.init_hidden(batch_size, inputs.device)
        else:
            init_h, init_c = hidden
            init_h = init_h.detach()  # 从计算图中分离，共享同一数据块，不再需要梯度
            init_c = init_c.detach()
            hidden = init_h.narrow(1, 0, batch_size), init_c.narrow(1, 0, batch_size)

        # rnn_out: (seq_len, batch_size, hidden_size * num_directions)
        # hidden_n: (h_n, c_n)  (num_layers * num_directions, batch_size, hidden_size)
        rnn_out, hidden = self.rnn_layer(embed, hidden)

        # (batch_size, hidden_size)
        h_n = hidden[0][-1]  # 以最后一层的末hidden state作为下一次的隐层输入
        if self.training:
            h_n = self.dropout_layer(h_n)

        # (batch_size, vocab_size)
        out = self.linear(h_n)

        return self.softmax(out), hidden

