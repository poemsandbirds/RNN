# -* coding:utf-8 *-

import numpy as np
from collections import namedtuple
from src.args import args

GRAD_NODE_FMT = namedtuple('grad_node', ['tensor', 'grad_fn'])
vocab_size = args.vocab_size
num_hiddens = args.num_hiddens


class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        if self.data.dtype == np.int and self.requires_grad:
            raise RuntimeError('Only Tensors of floating point and complex dtype can require gradients')
        self.grad_node = []
        self.grad = None

    def __repr__(self):
        s = 'tensor({}'.format(self.data)
        if self.grad_node:
            s += ', grad_fn=<{}>)'.format(self.grad_node[0].grad_fn.__name__)
        elif self.requires_grad:
            s += ', requires_grad=True)'
        else:
            s += ')'
        return s

    def __getitem__(self, item):
        data = self.data[item]
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def SelectBackward(grad):
                next_grad = np.zeros_like(self.data)
                next_grad[item] = grad
                return next_grad

            t.grad_node.append(GRAD_NODE_FMT(self, SelectBackward))

        return data

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        old_shape = self.data.shape
        t = Tensor(self.data.reshape(shape), self.requires_grad)

        if self.requires_grad:

            def ViewBackward(grad):
                grad = grad.reshape(old_shape)
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, ViewBackward))

        return t

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError('data does not require grad')
        if self.grad is None:
            if self.data.shape == () or self.data.shape == (1, ):
                self.grad = np.ones(1)
            else:
                raise RuntimeError('grad can be implicitly created only for scalar outputs')
        for node in self.grad_node:
            if node.tensor.grad is None:
                node.tensor.grad = node.grad_fn(self.grad)
            else:
                if node.tensor.shape != (num_hiddens, num_hiddens) and node.tensor.shape != (vocab_size, num_hiddens) \
                        and node.tensor.shape != (num_hiddens, vocab_size) and node.tensor.shape != (vocab_size,) \
                        and node.tensor.shape != (num_hiddens,):
                    node.tensor.grad = node.grad_fn(self.grad)
                else:
                    node.tensor.grad += node.grad_fn(self.grad)
            node.tensor.backward()

    def __add__(self, other):
        other = Tensor.astensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def AddBackward(grad):
                grad = grad * np.ones_like(self.data)
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(axis=0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == self.data.shape, 'AddBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, AddBackward))

        if other.requires_grad:

            def AddBackward(grad):
                grad = grad * np.ones_like(other.data)
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(axis=0)
                for i, d in enumerate(other.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == other.data.shape, 'AddBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(other, AddBackward))

        return t

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self - other

    def __neg__(self):
        data = - self.data
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if requires_grad:

            def NegBackward(grad):
                return -grad

            t.grad_node.append(GRAD_NODE_FMT(self, NegBackward))

        return t

    def __mul__(self, other):
        other = Tensor.astensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if requires_grad:

            def MulBackward(grad):
                grad = grad * other.data
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                assert grad.shape == self.data.shape, 'MulBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, MulBackward))

        if other.requires_grad:

            def MulBackward(grad):
                grad = grad * self.data
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                assert grad.shape == other.data.shape, 'MulBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(other, MulBackward))

        return t

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = Tensor.astensor(other)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def DivBackward(grad):
                grad = grad / other.data
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                assert grad.shape == self.data.shape, 'DivBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, DivBackward))

        if other.requires_grad:

            def DivBackward(grad):
                grad = - (self.data * grad) / (other.data**2)
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(other.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                assert grad.shape == other.data.shape, 'DivBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(other, DivBackward))

        return t

    def __matmul__(self, other):
        other = Tensor.astensor(other)

        if self.data.ndim == 1:
            self = self.reshape(1, -1)
        if other.data.ndim == 1:
            other = other.reshape(-1, 1)

        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def DotBackward(grad):
                grad = grad @ other.data.T
                assert grad.shape == self.data.shape, 'DotBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, DotBackward))

        if other.requires_grad:

            def DotBackward(grad):
                grad = self.data.T @ grad
                assert grad.shape == other.data.shape, 'DotBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(other, DotBackward))

        return t

    def ln(inputs):
        data = np.log(inputs.data)
        requires_grad = inputs.requires_grad
        t = Tensor(data, requires_grad)

        if inputs.requires_grad:

            def LogBackward(grad):
                return grad * (1/inputs.data)

            t.grad_node.append(GRAD_NODE_FMT(inputs, LogBackward))

        return t

    def tanh(inputs):
        data = np.tanh(inputs.data)
        requires_grad = inputs.requires_grad
        t = Tensor(data, requires_grad)

        if inputs.requires_grad:

            def TanhBackward(grad):
                return grad * (1 - np.tanh(inputs.data) ** 2)

            t.grad_node.append(GRAD_NODE_FMT(inputs, TanhBackward))

        return t

    def square(inputs):
        data = inputs.data ** 2
        requires_grad = inputs.requires_grad
        t = Tensor(data, requires_grad)

        if inputs.requires_grad:

            def SquareBackward(grad):
                return grad * (inputs.data*2)

            t.grad_node.append(GRAD_NODE_FMT(inputs, SquareBackward))

        return t

    def concatenate(inputs):

        def extract_data(X):
            Y = []
            for x in X:
                Y.append(x.data)
            return Y

        data = np.concatenate(extract_data(inputs), axis=0)
        t = Tensor(data, requires_grad=True)

        def CatBackward(i):
            def catbackward(grad):
                tmp = grad[i*32:i*32+32]
                return tmp
            return catbackward

        CatBackward_total = [CatBackward(i) for i in range(len(inputs))]

        for i in range(len(inputs)):
            t.grad_node.append(GRAD_NODE_FMT(inputs[i], CatBackward_total[i]))

        return t

    def loss_cross_entropy_softmax(self, other):
        other = Tensor.astensor(other)

        def softmax_func(x):
                row_max = x.max(axis=1)
                row_max = row_max.reshape(-1, 1)
                x = x - row_max
                x_exp = np.exp(x)
                x_sum = np.sum(x_exp, axis=1, keepdims=True)
                s = x_exp / x_sum
                return s

        data = np.zeros((self.shape[0], 1))
        log_softmax = - np.log(softmax_func(self.data))
        i = 0
        for j in range(self.shape[0]):
            data[i][0] = log_softmax[j] @ other.data[j].T
            i += 1
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:
            def CrossEntropyBackward(grad):
                next_grad = grad * (softmax_func(self.data) - other.data)
                return next_grad

            t.grad_node.append(GRAD_NODE_FMT(self, CrossEntropyBackward))

        return t

    def mean(self, dim=None, keepdim=False):
        data = self.data.mean(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def MeanBackward(grad):
                grad = grad * np.ones_like(self.data) / (self.data.reshape(-1).shape[0] / data.reshape(-1).shape[0])
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, MeanBackward))

        return t

    def sum(self, dim=None, keepdim=False):
        data = self.data.sum(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:

            def SumBackward(grad):
                return grad

            t.grad_node.append(GRAD_NODE_FMT(self, SumBackward))

        return t

    @staticmethod
    def astensor(data):
        if not isinstance(data, Tensor):
            data = Tensor(data)
        return data
