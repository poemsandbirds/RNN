# -* coding:utf-8 *-

import numpy as np
from src.args import args
from src.dataloader import data_loader
from src.autograd import Tensor
from train import RNNModel, train, predict


def get_lstm_params(
        vocab_size: int,
        num_hiddens: int
):
    """
    :param vocab_size: length of vocabulary
    :param num_hiddens: number of hidden layers
    :return: params for train
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens))

    def require_grad(x):
        tensor = Tensor(x, requires_grad=True)
        return tensor

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    params_output = []
    for param in params:
        params_output.append(require_grad(param))
    return params_output


def init_lstm_state(
        batch_size: int,
        num_hiddens: int
):
    """
    :return: initial state
    """
    return (np.zeros((batch_size, num_hiddens)),
            np.zeros((batch_size, num_hiddens)))


def lstm(
        inputs: np.ndarray,
        state: tuple,
        params: list
):
    """
    :return: output and next state
    """
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo,
     W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        X = Tensor(X)
        I = (X @ W_xi + H @ W_hi + b_i).sigmoid()
        F = (X @ W_xf + H @ W_hf + b_f).sigmoid()
        O = (X @ W_xo + H @ W_ho + b_o).sigmoid()
        C_tilda = (X @ W_xc + H @ W_hc + b_c).tanh()
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = H @ W_hq + b_q
        outputs.append(Y)
    outputs = Tensor.concatenate(outputs)
    return outputs, (H, C)


if __name__ == '__main__':
    batch_size, num_steps, num_hiddens = args.batch_size, args.num_steps, args.num_hiddens
    train_iter, vocab = data_loader(batch_size, num_steps)

    num_epochs, lr = 5, 1

    net = RNNModel(len(vocab), num_hiddens, get_lstm_params, init_lstm_state, lstm)

    print(predict('harry', 30, net, vocab))

    train(net, train_iter, vocab, lr, num_epochs)
