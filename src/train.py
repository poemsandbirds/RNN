# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader
from src.utils import onehot
import numpy as np
from src.timer import Timer
from src.accumulator import Accumulator
from src.autograd import Tensor

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
        outputs.append(Y)
    # return np.concatenate(outputs, axis=0), (H,)
    return outputs[0], (H,)

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
        y_origin = y['data']
        outputs.append(int(y_origin.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict('Harry', 10, net, vocab))


# if __name__ == '__init__':
