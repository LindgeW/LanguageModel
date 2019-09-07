import torch
import torch.nn as nn
import pickle
import numpy as np
import random
from datautil.dataloader import prepare_data


def testrand():
    x = list(range(100))
    w = x
    y = random.choices(x, weights=w, k=4)
    print(y)

def test1():
    token_lst = [['I', 'love', 'you', '!'],
                 ['Thanks', 'you', 'very', 'much'],
                 ['I', 'come', 'from', 'China', '!'],
                 ['I', 'love', 'eating', 'apple']]

    ctx_pairs = []
    pad = [''] * 3
    for tokens in token_lst:
        tokens = pad + tokens + pad
        for i in range(3, len(tokens)):
            ctx_pairs.append((tokens[i-3: i] + tokens[i+1: i+4], tokens[i]))
    print(ctx_pairs)


def test2():
    embed = nn.Embedding(10, 5)
    print(str(embed.weight.data.tolist()))
    # torch.save(embed.weight.data, './corpus/1.txt')
    with open('./corpus/1.pkl', 'wb') as fw:
        pickle.dump(embed.weight.data.tolist(), fw)
        # fw.writelines(map(str, embed.weight.data.tolist()))

    with open('./corpus/1.pkl', 'rb') as fin:
        ew = pickle.load(fin)
    print(ew)


def test3():
    embed = nn.Embedding(10, 5)

    embed_weights = embed.weight.data.numpy()
    print(embed_weights)
    np.save('./corpus/w.npy', embed_weights)

    em = np.load('./corpus/w.npy')
    print(em)


def test4():
    x = [[2, 3, 4, 5, 2], [5, 3, 6, 7, 9, 2], [34, 5, 7], [2, 46, 7, 4, 36]]
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 2:
                # del x[i][j]
                print(x[i][j])

def test5():
    documents = ["Tim bought a book .", "Tim is reading a book .",
                 "ah , Tim is Tim .", "I saw a boy reading a book ."]
    stop_words = ['a', '.', ',']

    clear_doc = []
    for doc in documents:
        clear_doc.append(' '.join([wd for wd in doc.split() if wd not in stop_words]))

    print(clear_doc)

def test6():
    dic = {"tom": 20, "jack": 21, "linda": 4, "peter": 12, "john": 12}
    print(max(dic))
    print(max(dic, key=dic.get))

    x = np.array([1, 9, 3, 5, 8, 0, 6])
    # x = np.random.randint(0, 20, (4, 5)).reshape(4, 5)
    print(x)
    # x.sort(axis=0)
    # print(x)
    print(x.argsort(axis=0))


def test7():
    bz = 5
    y = list('I come from china and I am proud of china.')
    print(y)
    max_len = len(y)
    offsets = [i*max_len//bz for i in range(bz)]
    print(max_len, offsets)

    iter = 0
    for i in range(bz):
        ctxt = [y[(of + iter) % max_len] for of in offsets]
        iter += 1
        tgt = [y[(of + iter) % max_len] for of in offsets]
        print(list(zip(ctxt, tgt)))


def test8():
    x = [list('I love china !'), list('Tom comes from America'), list('very good boy')]
    z = prepare_data(x, num_steps=4)
    for s in z:
        print(s)

def test9():
    x = torch.arange(30).reshape(1, 5, 6)
    print(x)
    # dim start length
    print(x.narrow(1, 1, 2))
    print(x.narrow(1, 0, 3))


from config.Const import *

def test10():
    x = [['i come from beijing'], ['good idea'], ['cool !']]
    print(x)
    for l in x:
        l.append(EOS)
    print(x)

def test11():
    embed = nn.Embedding(num_embeddings=5, embedding_dim=10)
    # x = torch.LongTensor([[1]])
    x = torch.LongTensor([1, 2, 3])
    out = embed(x)
    # print(out.shape)  # [1, 1, 10]
    print(out)

def test12():
    print(torch.tensor(1))
    print(torch.tensor([1]))
    print(torch.tensor(1).reshape(1))
    print(torch.tensor(1).unsqueeze(0))

    x = ['I', 'love', 'china']
    x.insert(0, '<bos>')
    x.append('xcv')
    print(x)

def test13():
    x = torch.arange(12).reshape(1, 3, 4)
    mask = torch.tensor([[1, 1, 1, 0],
                         [1, 1, 0, 0],
                         [1, 0, 0, 0]])
    print(x * mask)

def test14():
    x = torch.arange(12).reshape(3, 4)
    x.transpose_(0, 1)
    print(x)
    print(x[0])
    print(x[0].unsqueeze(1).repeat(1, 10))
    print(x[0].unsqueeze(1).expand(-1, 10))


def test15():
    x = torch.arange(24).reshape(3, 2, 4)
    print(x)
    print(x[:, -1, :])