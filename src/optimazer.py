# -* coding:utf-8 *-

import abc


class Optimizer(metaclass=abc.ABCMeta):

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.V = []
        self.m = []
        # for param in self.params:
        #     self.V.append(np.zeros_like(param.data))
        #     self.m.append(np.zeros_like(param.data))

    def zero_grad(self):
        for param in self.params:
            param.grad = 0
    @abc.abstractmethod
    def step(self):
        pass


class SGD(Optimizer):

    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad