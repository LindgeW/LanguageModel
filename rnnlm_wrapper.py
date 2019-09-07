import torch
import torch.optim as optim
import time 
import torch.nn.functional as F
from datautil.dataloader import data_iter


class RNNLMWrapper(object):
    def __init__(self, rnnlm, args, wd_vocab):
        super(RNNLMWrapper, self).__init__()
        self.rnnlm = rnnlm
        self.args = args
        self.wd_vocab = wd_vocab

    def summary(self):
        print(self.rnnlm)

    def train(self, train_data, valid_data):
        if self.args.optm == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.rnnlm.parameters()),
                                   lr=self.args.lr)
        else:
            # n-gram语言模型使用SGD效果更好！
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.rnnlm.parameters()),
                                  lr=self.args.lr,
                                  momentum=0.9,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=True)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(1 - epoch / self.args.epoch, 1e-4))

        for ep in range(self.args.epoch):
            self.rnnlm.train()

            start = time.time()
            lr_scheduler.step()

            train_loss = 0
            # (batch_size, seq_len)
            for word_idxs, mask in data_iter(train_data, self.args, self.wd_vocab, shuffle=True):
                hidden = None
                loss = 0
                # (seq_len, batch_size)
                word_idxs.transpose_(0, 1)
                mask.transpose_(0, 1)

                self.rnnlm.zero_grad()

                seq_len = word_idxs.size(0)
                lm_input = word_idxs[0]  # <bos>

                for i in range(1, seq_len):
                    pred, hidden = self.rnnlm(lm_input, hidden)
                    hidden *= mask[i].unsqueeze(1).repeat(1, hidden.size(-1))  # (1, 50, 200) * (50, 200)
                    loss += self._calc_loss(pred, word_idxs[i])
                    lm_input = word_idxs[i]

                train_loss += loss.data.item()
                # 第一次反向传播之后，计算图的内存就会被释放掉，这样的话再次进行反向传播就不行了
                loss.backward()
                optimizer.step()

            end = time.time()
            print('[Epoch %d] train loss: %.3f' % (ep + 1, train_loss / len(train_data)))
            val_loss = self.validate(valid_data)
            print('dev loss: %.3f' % val_loss)
            print('lr: ', lr_scheduler.get_lr())
            print('time cost: %.2fs' % (end - start))

    def validate(self, val_data):
        self.rnnlm.eval()

        val_loss = 0
        with torch.no_grad():
            for word_idxs, mask in data_iter(val_data, self.args, self.wd_vocab):
                hidden = None
                # (seq_len, batch_size)
                word_idxs.transpose_(0, 1)
                mask.transpose_(0, 1)

                seq_len = word_idxs.size(0)
                lm_input = word_idxs[0]  # <bos>
                for i in range(1, seq_len):
                    pred, hidden = self.rnnlm(lm_input, hidden)
                    hidden *= mask[i].unsqueeze(1).repeat(1, hidden.size(-1))  # (1, 50, 200) * (50, 200)
                    val_loss += self._calc_loss(pred, word_idxs[i]).data.item()
                    lm_input = word_idxs[i]

        return val_loss / len(val_data)

    def evaluate(self, test_data):
        self.rnnlm.eval()

        test_loss = 0
        with torch.no_grad():
            for word_idxs, mask in data_iter(test_data, self.args, self.wd_vocab):
                hidden = None
                # (seq_len, batch_size)
                word_idxs.transpose_(0, 1)
                mask.transpose_(0, 1)

                seq_len = word_idxs.size(0)
                lm_input = word_idxs[0]  # <bos>
                for i in range(1, seq_len):
                    pred, hidden = self.rnnlm(lm_input, hidden)
                    hidden *= mask[i].unsqueeze(1).repeat(1, hidden.size(-1))  # (1, 50, 200) * (50, 200)
                    test_loss += self._calc_loss(pred, word_idxs[i]).data.item()
                    lm_input = word_idxs[i]

        test_loss /= len(test_data)
        print('test loss: %.3f' % test_loss)
        return test_loss

    def _calc_loss(self, pred, tgt):
        return F.nll_loss(pred, tgt, ignore_index=0)
