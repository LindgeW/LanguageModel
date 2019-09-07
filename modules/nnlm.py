'''
    基于n-gram的神经语言模型 (context, next_word)
'''
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class LanguageModel(nn.Module):
    def __init__(self, args):
        super(LanguageModel, self).__init__()

        self.embed = nn.Embedding(num_embeddings=args.vocab_size,
                                  embedding_dim=args.word_size,
                                  padding_idx=0)

        nn.init.xavier_uniform_(self.embed.weight)

        self._drop_embed = nn.Dropout(args.drop_embed)

        self._mlp = nn.Sequential(
            nn.Linear(in_features=args.ctx_size * args.word_size,
                      out_features=args.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=args.hidden_size,
                      out_features=args.vocab_size)
        )

    def save_embed_weights(self, save_path):
        # with open(save_path, 'wb') as fw:
        #     pickle.dump(self.embed.weight.data.tolist(), fw)

        # numpy保存
        np.save(save_path, self.embed.weight.data.numpy())

    def load_embed_weights(self, load_path):
        # with open(load_path, 'wb') as fin:
        #     embed_weights = pickle.load(fin)

        embed_weights = np.load(load_path)
        return embed_weights

    def forward(self, ctx_inputs):
        '''
        :param ctx_inputs:  (bz, ctx_size)
        :return:
        '''

        batch_size = ctx_inputs.size(0)

        # (bz, ctx_size, word_size)
        embed = self.embed(ctx_inputs)

        if self.training:
            embed = self._drop_embed(embed)

        # (bz, ctx_size*word_size) -> (bz, vocab_size)
        out = self._mlp(embed.reshape(batch_size, -1))

        # 计算量很大，需要做加速优化-负采样、层序softmax
        return F.log_softmax(out, dim=-1)
