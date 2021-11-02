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

batch_size, num_steps, num_hiddens = args.batch_size, 5, args.num_hiddens
train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)


# 初始化参数
def get_params(num_steps, num_hiddens):
    num_inputs = num_outputs = num_steps

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
#
# def get_params_txt():
#     with open ('../output/params.txt','r') as f:
#         lines = f.read()
#         lines = np.array(float(line) for line in lines)
#         print(lines)

# a = np.loadtxt('../output/params')
# print(a)
# exit()

def load_params():
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
    params = [W_xh,W_hh,b_h,W_hq,b_q]
    return params
params = get_params(num_steps, num_hiddens)
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
        H = Tensor(H)
    outputs = []
    X = inputs
    X = Tensor(X, requires_grad=False)
    H = (X @ W_xh + H @ W_hh + b_h).tanh()
    Y = H @ W_hq + b_q
    return Y, (H,)
    # return outputs[0], (H,)


# 定义一个类来封装之前的函数
class RNNModel:

    def __init__(self, num_steps, num_hiddens, get_params,
                 init_state, forward_fn):
        self.num_steps, self.num_hiddens = num_steps, num_hiddens
        # self.params = get_params(num_steps, num_hiddens)
        self.params = get_params()
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)


net = RNNModel(num_steps=num_steps, num_hiddens=num_hiddens, get_params=load_params,
               init_state=init_rnn_state, forward_fn=rnn)
# print(net.params[0].shape)


def mse(y_hat, y):
    loss = ((y_hat - y).square())/2
    return loss


def predict(prefix, num_preds, net, vocab):
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]],vocab[prefix[1]],vocab[prefix[2]],vocab[prefix[3]],vocab[prefix[4]]]
    outputs = np.array(outputs)
    # print(outputs)
    for i in range(num_preds-5):
        y, state = net(outputs[-5:,], state)
        outputs = np.append(outputs, y.data[-1])
    outputs = outputs.astype(int)
    # print(outputs)
    outputs_sentence = ''.join([vocab.idx_to_token[i] for i in outputs])
    return outputs_sentence



print(predict('Harry', 50, net, vocab))
# exit()


# def cross_entropy(y_hat, y):
#     y_softmax = y_hat.softmax()
#     # y_softmax.mean().backward()
#     y = y.astype(int)
#     y = Tensor(y, requires_grad=False)
#     tmp = y_softmax.replace(y)
#     loss = -(tmp.log())
#     # return -np.log(y_hat[range(len(y_hat)), y])
#     return loss


def grad_clipping(net, theta):
    params = net.params
    # print(p.grad for p in params)
    # print(sum((p.grad ** 2) for p in params))
    # exit()
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, loss, updater):
    state, timer = None, Timer()
    # metric = Accumulator(2)
    # 训练损失之和, 词元数量
    loss1 = 0.
    for X, Y in train_iter:
        Y = Y.astype(float)
        state = net.begin_state(batch_size=X.shape[0])
        Y_hat, state = net(X, state)
        l = loss(Y_hat, Y).mean()
        losss = l.data
        updater.zero_grad()
        l.backward()
        # print(net.params[0].grad)
        grad_clipping(net, 1)
        updater.step()
        loss1 += losss
        # print(11)
        # metric.add(l * y.size, y.size)
    # return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
    return loss1

def train(net, train_iter, vocab, lr, num_epochs):
    loss = mse
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # 初始化
    updater = SGD(net.params, lr)
    predict = lambda prefix: predict(prefix, 50, net, vocab)
    # 训练和预测
    i = 0
    for epoch in range(num_epochs):
        l = train_epoch(net, train_iter, loss, updater)
        print(l)
        i +=1
        print(i)
        # print(net.params)
        # print(predict('time traveller'))
        # if (epoch + 1) % 10 == 0:
        #     animator.add(epoch + 1, [ppl])
    # print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))
    # print(predict('Harry Potter', 50, net, vocab))

lr = 1
num_epochs = 10
# print(params[0].grad)
net = RNNModel(num_steps=5, num_hiddens=num_hiddens, get_params=load_params,
               init_state=init_rnn_state, forward_fn=rnn)
train(net, train_iter, vocab, lr, num_epochs)
print(predict('Harry', 50, net, vocab))

# print(net.params[4])
# exit()
# net.params[0].data.tolist()
# net.params[1].data.tolist()
# net.params[2].data.tolist()
# net.params[3].data.tolist()
# net.params[4].data.tolist()
#
# with open ('../output/params.txt', 'w') as f:
#     f.write(str(net.params[0].data))
#     f.write('\n')
#     f.write(str(net.params[1].data))
#     f.write('\n')
#     f.write(str(net.params[2].data))
#     f.write('\n')
#     f.write(str(net.params[3].data))
#     f.write('\n')
#     f.write(str(net.params[4].data))
np.savetxt('../output/params_0', net.params[0].data)
np.savetxt('../output/params_1', net.params[1].data)
np.savetxt('../output/params_2', net.params[2].data)
np.savetxt('../output/params_3', net.params[3].data)
np.savetxt('../output/params_4', net.params[4].data)
# if __name__ == '__init__':
