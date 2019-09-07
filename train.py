'''
基于n-gram语言模型生产词的特征表示
    1、加载语料(一行一句，单行每词隔开)
    2、构建词表
    3、搭建语言模型(MLP -> LSTM)
    4、训练
    5、导出嵌入层的词向量
'''
import numpy as np
import torch
import torch.nn as nn
from config.Config import data_path, arg_conf
from datautil.dataloader import *
from vocab.Vocab import build_vocab
# from modules.nnlm import LanguageModel
from modules.rnnlm import *
from lm_wrapper import Wrapper


if __name__ == '__main__':
    # 随机种子
    np.random.seed(1234)
    torch.manual_seed(1314)
    torch.cuda.manual_seed(3347)

    # 配置参数
    data_path = data_path('config/data_path.json')
    print(data_path)
    args = arg_conf()
    args.weight_path = data_path['weight_path']['save']

    print('GPU available:', torch.cuda.is_available())
    print('cuDNN:', torch.backends.cudnn.enabled)
    print('GPU number:', torch.cuda.device_count())

    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    # 加载数据
    train_raw_data = load_data(data_path['data']['train'])
    valid_raw_data = load_data(data_path['data']['valid'])
    test_raw_data = load_data(data_path['data']['test'])

    print(f'raw data lines: {len(train_raw_data)}')
    # train_data = produce_ngram(train_raw_data, args.ctx_size)
    train_data = prepare_data(train_raw_data, args.ctx_size)
    valid_data = prepare_data(valid_raw_data, args.ctx_size)
    test_data = prepare_data(test_raw_data, args.ctx_size)
    print(f'train data size: {len(train_data)}')

    # 构建词表
    wd_vocab = build_vocab(data_path['data']['train'], args.min_count)
    args.vocab_size = wd_vocab.vocab_size

    # 构建模型
    # nnlm = LanguageModel(args).to(args.device)
    rnnlm = RNNLanguageModel(args).to(args.device)

    # if torch.cuda.device_count() > 1:
    #     nnlm = nn.DataParallel(nnlm, device_ids=[0, 1])

    wrapper = Wrapper(rnnlm, args, wd_vocab)
    wrapper.summary()

    # 训练
    wrapper.train(train_data, valid_data)

    # 评估
    wrapper.evaluate(test_data)

