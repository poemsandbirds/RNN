# -* coding:utf-8 *-

from src.args import args
import collections
import random
import re
import numpy as np


data_path = args.data_path


def read_data(
):
    """
    :return: read data, strip symbol and make letters lower
    """
    with open(data_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(
        lines
):
    """
    :return: list
    """
    return [list(line) for line in lines]


def count_corpus(
        tokens: list
):
    """
    :return: counter frequency of tokens
    """
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def data_preprocess(
):
    """
    :return: corpus and vocab
    """
    lines = read_data()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return corpus, vocab


def seq_data_iter_random(
        corpus,
        batch_size,
        num_steps
):
    """
    :return: use random method to separate data into sequence
    """
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)


def data_loader(
        batch_size,
        num_steps
):
    """
    :return:  data iterator and vocab for predict
    """
    data_iter = SeqDataLoader(batch_size, num_steps)
    return data_iter, data_iter.vocab


class Vocab:

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


class SeqDataLoader:

    def __init__(self, batch_size, num_steps):
        self.data_iter_fn = seq_data_iter_random
        self.corpus, self.vocab = data_preprocess()
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
