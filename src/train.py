# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader
from src.utils import onehot
import numpy as np

batch_size, num_steps = args.batch_size, args.num_steps
train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)


"""初始化参数"""
def get_params(vocab_size, num_hiddens):
    # num_hiddens是隐藏层参数个数
    num_inputs = num_outputs = vocab_size
    def normal(shape1, shape2):
        return np.random.randn(shape1, shape2) * 0.01
    # 隐藏层参数
    W_xh = normal(num_inputs, num_hiddens)
    W_hh = normal(num_hiddens, num_hiddens)
    b_h = np.zeros(num_hiddens)
    # 输出层参数
    W_hq = normal(num_hiddens, num_outputs)
    b_q = np.zeros(num_outputs)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    # for param in params:
    #     param.requires_grad_(True)
    return params

params = get_params(len(vocab), 100)
print(params)
# if __name__ == '__init__':
