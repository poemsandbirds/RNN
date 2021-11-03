# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader
from src.train import RNNModel, load_params, init_rnn_state, rnn, predict


if __name__ == '__main__':
        batch_size, num_steps, num_hiddens = args.batch_size, args.num_steps, args.num_hiddens
        train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)

        net = RNNModel(len(vocab), num_hiddens, load_params, init_rnn_state, rnn)

        print(predict('harry', 500, net, vocab))
