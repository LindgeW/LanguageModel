import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datautil.dataloader import batch_iter
import time


class Wrapper(object):
    def __init__(self, nnlm, args, vocab):
        assert isinstance(nnlm, nn.Module)
        self.nnlm = nnlm
        self.args = args
        self.wd_vocab = vocab

    def summary(self):
        print(self.nnlm)

    # RNN LM 训练
    def train(self, train_data, valid_data):
        if self.args.optm == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.nnlm.parameters()),
                                   lr=self.args.lr)
        else:
            # n-gram语言模型使用SGD效果更好！
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.nnlm.parameters()),
                                  lr=self.args.lr,
                                  momentum=0.9,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=True)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(1 - epoch/self.args.epoch, 1e-4))

        # optimizer = nn.DataParallel(optimizer, device_ids=[0, 1])

        for ep in range(self.args.epoch):
            self.nnlm.train()

            start = time.time()
            train_loss = 0
            hidden = None
            lr_scheduler.step()
            for ctx, tgt in batch_iter(train_data, self.args, self.wd_vocab, shuffle=False):
                self.nnlm.zero_grad()
                pred, hidden = self.nnlm(ctx, hidden)
                loss = self._calc_loss(pred, tgt)
                train_loss += loss.data.item()
                # 第一次反向传播之后，计算图的内存就会被释放掉，这样的话再次进行反向传播就不行了
                loss.backward()
                optimizer.step()
                # optimizer.module.step()  # optimizer在DataParallel中，需要变成普通的Adam才能调用step

            end = time.time()
            print('[Epoch %d] train loss: %.3f' % (ep+1, train_loss / len(train_data)))
            val_loss = self.validate(valid_data)
            print('dev loss: %.3f' % val_loss)
            print('lr: ', lr_scheduler.get_lr())
            print('time cost: %.2fs' % (end - start))

        # self.save_word_vec(self.args.weight_path)

    def validate(self, valid_data):
        val_loss = 0
        self.nnlm.eval()
        with torch.no_grad():
            for ctx, tgt in batch_iter(valid_data, self.args, self.wd_vocab):
                pred, _ = self.nnlm(ctx)
                loss = self._calc_loss(pred, tgt)
                val_loss += loss.data.item()

        val_loss /= len(valid_data)

        return val_loss

    def evaluate(self, test_data):
        test_loss = 0
        self.nnlm.eval()
        with torch.no_grad():
            for ctx, tgt in batch_iter(test_data, self.args, self.wd_vocab):
                pred, _ = self.nnlm(ctx)
                loss = self._calc_loss(pred, tgt)
                test_loss += loss.data.item()
        print('test data loss: %.3f' % (test_loss / len(test_data)))

    def _calc_loss(self, pred, target):
        return F.nll_loss(pred, target)

    def save_word_vec(self, path):
        assert os.path.exists(path)
        self.nnlm.save_embed_weights(path)
