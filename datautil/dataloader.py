import os
import torch
import numpy as np
from config.Const import *


class Instance(object):
    def __init__(self, ctx_wds, goal_wd):
        self.ctx_wds = ctx_wds      # 上下文
        self.goal_wd = goal_wd      # 待预测词

    def __str__(self):
        return str(self.ctx_wds) + ' ' + str(self.goal_wd)


# 加载语料库中的数据
def load_data(path):
    assert os.path.exists(path)

    token_lst = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line != '':
                token_lst.append(line.strip().split(' '))
        return token_lst


# 将数据转换成n-gram形式：(context words, central word)
# 1、只考虑上文，预测下一个词
def produce_ngram(token_lst, ctx_size, pad='<pad>'):
    insts = []
    pad = [pad] * ctx_size
    for tokens in token_lst:
        tokens = pad + tokens
        for i in range(ctx_size, len(tokens)):
            inst = Instance(tokens[i-ctx_size: i], tokens[i])
            insts.append(inst)
    return insts


# 2、考虑上下文，预测中心词
# def produce_ngram(token_lst, ctx_size):
#     insts = []
#     pad = ['<pad>'] * ctx_size
#     for tokens in token_lst:
#         tokens = pad + tokens + pad
#         for i in range(len(tokens)-ctx_size):
#             inst = Instance(tokens[i-ctx_size: i] + tokens[i+1: i+ctx_size+1], tokens[i])
#             insts.append(inst)
#     return insts


# 每个序列截断定长num_steps个词，下一个词作为预测目标
def prepare_data(token_lst, num_steps, pad=PAD):
    insts = []
    for tokens in token_lst:
        for i in range(len(tokens)):
            if i < num_steps:
                ctx_seq = [pad] * (num_steps - i) + tokens[:i]
                tgt = tokens[i]
            else:
                ctx_seq = tokens[i-num_steps: i]
                tgt = tokens[i]

            insts.append(Instance(ctx_seq, tgt))
    return insts


def batch_iter(dataset, args, wd_vocab, shuffle=True):
    if shuffle:
        np.random.shuffle(dataset)

    batch_size = args.batch_size
    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i * batch_size: (i+1) * batch_size]
        yield batch_gen(batch_data, wd_vocab, args.ctx_size, args.device)


def batch_gen(batch_data, wd_vocab, ctx_size, device):
    batch_size = len(batch_data)
    ctx_idxs = torch.zeros((batch_size, ctx_size), dtype=torch.long, device=device)
    goal_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, inst in enumerate(batch_data):
        ctx_idxs[i, :] = torch.tensor(wd_vocab.word2index(inst.ctx_wds), device=device)
        goal_idxs[i] = torch.tensor(wd_vocab.word2index(inst.goal_wd), device=device)

    return ctx_idxs, goal_idxs


# ====================================================================== #

# 在每句话开头和结尾分别添加bos和eos
def prepare_dataset(token_lst, start_token=BOS, end_token=EOS):
    for tokens in token_lst:
        tokens.insert(0, start_token)
        tokens.append(end_token)

    return token_lst


def data_iter(dataset, args, wd_vocab, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    batch_size = args.batch_size
    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        # 一个batch里面的最大序列长度
        max_seq_len = max(len(tokens) for tokens in batch_data)
        wd_idxs = torch.zeros((len(batch_data), max_seq_len), dtype=torch.long, device=args.device)
        non_pad_mask = torch.zeros((len(batch_data), max_seq_len), device=args.device)
        for j, line in enumerate(batch_data):
            seq_len = len(line)
            wd_idxs[j, :seq_len] = torch.tensor(wd_vocab.word2index(line), device=args.device)
            non_pad_mask[j, :seq_len].fill_(1)
        # 高效生成mask
        # non_pad_mask = wd_idxs.ne(wd_vocab.PAD).float()

        yield wd_idxs, non_pad_mask
