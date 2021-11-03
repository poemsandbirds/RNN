# -* coding:utf-8 *-


class Optimizer():

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    def step(self):
        pass


class SGD(Optimizer):

    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad