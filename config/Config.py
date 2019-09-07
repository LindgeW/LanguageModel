from argparse import ArgumentParser
from datautil.jsonutil import JsonUtil


# 加载数据参数
def data_path(file_path):
    return JsonUtil.file_loader(file_path)


# 命令行参数解析
def arg_conf():
    parser = ArgumentParser('Language Model Configure')
    parser.add_argument('--cuda', type=int, default=-1, help='training device, default on cpu')

    parser.add_argument('--min_count', type=int, default=5, help='low frequent word if less than min_count')
    parser.add_argument('--word_size', type=int, default=100, help='word vector size')

    # 上下文序列长度大小
    parser.add_argument('--ctx_size', type=int, default=20, help='the word number of context')
    # 截取的序列长度
    # parser.add_argument('--num_steps', type=int, default=20, help='the length of truncate sequence')

    parser.add_argument('--hidden_size', type=int, default=200, help='the hidden size of network')
    parser.add_argument('--num_layers', type=int, default=1, help='the number of rnn layers')

    parser.add_argument('--epoch', type=int, default=20, help='the number of training iter')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--drop_embed', type=float, default=0.5, help='dropout rate of embedding layer')
    parser.add_argument('--drop_rnn', type=float, default=0., help='dropout rate of rnn layer')

    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of network')
    parser.add_argument('--optm', type=str, default='SGD', help='the optimizer of update gradient')

    args = parser.parse_args()
    print(vars(args))
    return args

