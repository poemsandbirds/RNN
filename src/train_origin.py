# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader
from src.utils import onehot
import numpy as np
from src.accumulator import Accumulator
from src.autograd import Tensor
from src.optimazer import SGD
import math


"""
    存在很大问题，写成全连接层了
"""


def get_params(
        vocab_size: int,
        num_hiddens: int
):
    """
    :param vocab_size: how many vocab in book
    :param num_hiddens: number of parameter of hidden layers
    :return: params
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape)

    W_xh = normal((num_inputs, num_hiddens))
    W_xh = Tensor(W_xh, requires_grad=True)
    W_hh = normal((num_hiddens, num_hiddens))
    W_hh = Tensor(W_hh, requires_grad=True)
    b_h = np.zeros(num_hiddens)
    b_h = Tensor(b_h, requires_grad=True)
    W_hq = normal((num_hiddens, num_outputs))
    W_hq = Tensor(W_hq, requires_grad=True)
    b_q = np.zeros(num_outputs)
    b_q = Tensor(b_q, requires_grad=True)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def load_params(
        vocab_size,
        num_hiddens
):
    """
    :return: params from local
    """
    W_xh = np.loadtxt('../output/params_0')
    W_hh = np.loadtxt('../output/params_1')
    b_h = np.loadtxt('../output/params_2')
    W_hq = np.loadtxt('../output/params_3')
    b_q = np.loadtxt('../output/params_4')
    W_xh = Tensor(W_xh, requires_grad=True)
    W_hh = Tensor(W_hh, requires_grad=True)
    b_h = Tensor(b_h, requires_grad=True)
    W_hq = Tensor(W_hq, requires_grad=True)
    b_q = Tensor(b_q, requires_grad=True)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(
        batch_size: int,
        num_hiddens: int,
        num_steps: int
):
    """
    :return: state to init
    """
    H = np.zeros((batch_size * num_steps, num_hiddens), dtype=float)
    return (H,)


def rnn(
        inputs: np.ndarray,
        state: tuple,
        params: list
):
    """
    :param inputs: input
    :param state: hidden state
    :return: output
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    H = Tensor.astensor(H)
    X = Tensor(inputs.reshape(inputs.shape[0] * inputs.shape[1], 27))
    H = (X @ W_xh + H @ W_hh + b_h).tanh()
    output = H @ W_hq + b_q
    return output, (H,)


class RNNModel:

    def __init__(self, vocab_size, num_hiddens, get_params, num_steps,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens, self.num_steps = vocab_size, num_hiddens, num_steps
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = onehot(X.T, self.vocab_size).astype(float)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens, self.num_steps)


def predict(
        prefix: str,
        num_preds: int,
        net,
        vocab
):
    """
    :param prefix: inputs of sentence
    :param num_preds: numbers of predictions
    :param net: net
    :param vocab: vocab
    :return: predicted sentence
    """
    state = net.begin_state(batch_size=1)
    outputs = [int(vocab[prefix[i]]) for i in range(5)]
    get_input = lambda: np.array([outputs[-5], outputs[-4], outputs[-3], outputs[-2], outputs[-1]], dtype=float).reshape((1, 5))
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(np.argmax(y.data, axis=1)[4])
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(
        net,
        theta: float
):
    """
    :param net: net
    :param theta: parameter to control gradient
    """
    params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
                param.grad *= theta / norm


def train_epoch(
        net,
        train_iter,
        updater
):
    """
    :return: train an epoch and return loss
    """
    metric = Accumulator(1)
    i = 0
    for X, Y in train_iter:
        Y = Y.astype(float)
        state = net.begin_state(batch_size=X.shape[0])
        Y = onehot(Y.T, len(vocab)).astype(float)
        tmp = []
        for t in Y:
            tmp.append(t)
        Y = np.concatenate(tmp, axis=0)
        Y = Tensor(Y, requires_grad=False)
        y_hat, state = net(X, state)
        l = y_hat.loss_cross_entropy_softmax(Y).mean()
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        metric.add(l.data)
        print(i)
        i += 1
    return metric[0]


def train(
        net,
        train_iter,
        vocab,
        lr: float,
        num_epochs: int
):
    """
    :param lr: learning rate
    :param num_epochs: number of epochs
    """
    updater = SGD(net.params, lr)
    for epoch in range(num_epochs):
        l = train_epoch(net, train_iter, updater)
        print('loss:{}'.format(l))
        print(predict('harry', 50, net, vocab))


if __name__ == '__main__':
    batch_size, num_steps, num_hiddens = args.batch_size, 5, args.num_hiddens
    train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)

    net = RNNModel(len(vocab), num_hiddens, get_params, num_steps, init_rnn_state, rnn)

    print(predict('harry', 50, net, vocab))

    lr = 1
    num_epochs = 10

    train(net, train_iter, vocab, lr, num_epochs)

    np.savetxt('../output/params_0', net.params[0].data)
    np.savetxt('../output/params_1', net.params[1].data)
    np.savetxt('../output/params_2', net.params[2].data)
    np.savetxt('../output/params_3', net.params[3].data)
    np.savetxt('../output/params_4', net.params[4].data)
