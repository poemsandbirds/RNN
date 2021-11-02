# -* coding:utf-8 *-

from src.args import args
import collections
import random
import re
import numpy as np

data_path = args.data_path


def read_data():
    with open(data_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines):
    return [list(line) for line in lines]


# 返回一个带有字符频率的字典
def count_corpus(tokens):
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表,将字符根据频率对应索引"""
    def __init__(self, tokens):
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        uniq_tokens = []
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq > 0]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        return self.token_to_idx.get(tokens)

    def to_tokens(self, indices):
        return self.idx_to_token[indices]


def data_preprocess():
    lines = read_data()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机抽样生成一个小批量的子序列。"""
    # 先根据字符串长度分区再打乱
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)
    def data(pos):
        # 返回从`pos`位置开始的长度为`num_steps`的序列
        return corpus[pos:pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)


class SeqDataLoader:
    """加载序列数据的迭代器。"""
    def __init__(self, batch_size, num_steps):
        self.data_iter_fn = seq_data_iter_random
        self.corpus, self.vocab = data_preprocess()
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def data_loader(batch_size, num_steps):
    """返回迭代器(用以训练)和词汇表。"""
    data_iter = SeqDataLoader(batch_size, num_steps)
    return data_iter, data_iter.vocab