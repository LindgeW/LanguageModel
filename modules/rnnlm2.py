import torch.nn as nn
import torch
import numpy as np
from .rnn_encoder import RNNEncoder

'''
方式二：按照时间步，逐步预测，累积误差
'''


class RNNLM(nn.Module):
    def __init__(self, args):
        super(RNNLM, self).__init__()

        self.args = args

        self.hidden_size = args.hidden_size

        # 副产品
        self.embed_layer = nn.Embedding(num_embeddings=args.vocab_size,
                                        embedding_dim=args.word_size,
                                        padding_idx=0)

        nn.init.xavier_uniform_(self.embed_layer.weight)

        # 不定长序列需要pack-pad sequence
        self.rnn_layer = nn.GRU(
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
        #     bidirectional=False,
        #     rnn_type='gru'
        # )

        self.dropout_embed = nn.Dropout(args.drop_embed)
        self.dropout_layer = nn.Dropout(args.drop_rnn)

        self.linear = nn.Linear(in_features=args.hidden_size,
                                out_features=args.vocab_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size=1, device=torch.device('cpu')):
        h0 = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        # h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0

    def save_embed_weights(self, path):
        np.save(path, self.embed_layer.weight.data.numpy())

    # 每次只处理一个词
    def forward(self, inputs, hidden=None):
        '''
        :param inputs: (batch_size, 1)
        :param hidden: (batch_size, 1, hidden_size)
        :return:
        '''
        batch_size = inputs.shape[0]

        # (batch_size, 1) -> (batch_size, 1, embed_dim)
        embed = self.embed_layer(inputs).unsqueeze(1)

        if self.training:
            embed = self.dropout_embed(embed)

        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size, device=inputs.device)

        # rnn_out: (seq_len, batch_size, hidden_size * num_directions)
        # hidden_n: (h_n, c_n)  (num_layers * num_directions, batch_size, hidden_size)
        rnn_out, hidden = self.rnn_layer(embed, hidden)

        # (batch_size, 1, hidden_size) -> (batch_size, 1, vocab_size)
        out = self.linear(rnn_out).squeeze(1)

        return self.softmax(out), hidden
