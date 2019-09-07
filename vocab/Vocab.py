import os
from datautil.dataloader import load_data
from collections import Counter
from config.Const import *


def build_vocab(path, min_count):
    wd_counter = Counter()
    token_lst = load_data(path)
    for tokens in token_lst:
        wd_counter.update(tokens)
    return WordVocab(wd_counter, min_count)


class WordVocab(object):
    def __init__(self, wd_counter, min_count=5):
        super(WordVocab, self).__init__()
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3

        self._wd2freq = dict(filter(lambda x: x[1] >= min_count, wd_counter.items()))
        # self._wd2freq = dict((wd, freq) for wd, freq in wd_counter.items() if freq >= min_count)

        self._wd2idx = {
                        PAD: self.PAD,
                        UNK: self.UNK,
                        BOS: self.BOS,
                        EOS: self.EOS
                    }

        for wd in self._wd2freq.keys():
            if wd not in self._wd2idx:
                self._wd2idx[wd] = len(self._wd2idx)

        self._idx2wd = dict((idx, wd) for wd, idx in self._wd2idx.items())

        print(f'vocab size: {self.vocab_size}')

    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._wd2idx.get(wd, self.UNK) for wd in wds]
        else:
            return self._wd2idx.get(wds, self.UNK)

    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i) for i in idxs]
        else:
            return self._idx2wd.get(idxs)

    @property
    def vocab_size(self):
        return len(self._wd2idx)

    # 保存词频表
    def save_freq_vocab(self, path):
        assert os.path.exists(path)
        with open(path, 'a', encoding='utf-8') as fw:
            for wd, freq in self._wd2freq.items():
                fw.write(f'{wd} {freq}')
                fw.write('\n')
