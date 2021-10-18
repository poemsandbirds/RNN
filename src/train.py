# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader
from src.utils import onehot
import numpy as np
from src.timer import Timer
from src.accumulator import Accumulator
from src.autograd import Tensor
from src.optimazer import SGD
import math

batch_size, num_steps, num_hiddens = args.batch_size, args.num_steps, args.num_hiddens
train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)


# 初始化参数
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape)

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_xh = Tensor(W_xh, requires_grad=True)
    W_hh = normal((num_hiddens, num_hiddens))
    W_hh = Tensor(W_hh, requires_grad=True)
    b_h = np.zeros(num_hiddens)
    b_h = Tensor(b_h, requires_grad=True)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    W_hq = Tensor(W_hq, requires_grad=True)
    b_q = np.zeros(num_outputs)
    b_q = Tensor(b_q, requires_grad=True)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


params = get_params(len(vocab), num_hiddens)
# print(params)


# 循环神经网络模型
# 初始化隐藏状态H
def init_rnn_state(batch_size, num_hiddens):
    return (np.zeros((batch_size, num_hiddens), dtype=float), )


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    if isinstance(H, Tensor):
        H = H
    else:
        H = Tensor(H, requires_grad=True)
    outputs = []
    for X in inputs:
        X = Tensor(X, requires_grad=True)
        H = (X @ W_xh + H @ W_hh + b_h).tanh()
        Y = H @ W_hq + b_q
        Y_origin = Y['data']
        outputs.append(Y_origin)
    return np.concatenate(outputs, axis=0), (H,)
    # return outputs[0], (H,)


# 定义一个类来封装之前的函数
class RNNModel:

    def __init__(self, vocab_size, num_hiddens, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = onehot(X.T, self.vocab_size).astype(float)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)


net = RNNModel(len(vocab), num_hiddens, get_params,
               init_rnn_state, rnn)


def predict(prefix, num_preds, net, vocab):
    state = net.begin_state(batch_size=1)
    outputs = [int(vocab[prefix[0]])]
    get_input = lambda: np.array([outputs[-1]], dtype=float).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(int(vocab[y]))
        # print(outputs)
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # print(y.shape)
        # y_origin = y['data']
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict('Harry', 10, net, vocab))


def cross_entropy(y_hat, y):
    # print(y, y_hat)
    y_hat = Tensor(y_hat, requires_grad=True)
    y_hat = y_hat.softmax()
    y = Tensor(y, requires_grad=True)
    # print(y)
    tmp = y_hat[range(len(y_hat)), y]
    tmp = Tensor(tmp, requires_grad=True)
    loss = -(tmp).log()
    # return -np.log(y_hat[range(len(y_hat)), y])
    return loss


def grad_clipping(net, theta):
    params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, loss, updater):
    state, timer = None, Timer()
    # metric = Accumulator(2)
    # 训练损失之和, 词元数量
    for X, Y in train_iter:
        Y = Y.astype(float)
        state = net.begin_state(batch_size=X.shape[0])
        y = Y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        updater.zero_grad()
        l.backward()
        # grad_clipping(net, 1)
        updater.step()
        # metric.add(l * y.size, y.size)
    # return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs):
    loss = cross_entropy
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # 初始化
    updater = SGD(net.params, lr)
    predict = lambda prefix: predict(prefix, 50, net, vocab)
    # 训练和预测
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, updater)
        print(1)
        # print(predict('time traveller'))
        # if (epoch + 1) % 10 == 0:
        #     animator.add(epoch + 1, [ppl])
    # print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))
    # print(predict('Harry Potter', 50, net, vocab))

lr = 1
num_epochs = 10
train(net, train_iter, vocab, lr, num_epochs)
print(predict('Harry', 50, net, vocab))


# if __name__ == '__init__':
