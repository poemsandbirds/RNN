# -* coding:utf-8 *-

from src.args import args
from src.dataloader import data_loader

batch_size, num_steps = args.batch_size, args.num_steps
train_iter, vocab = data_loader(batch_size=batch_size, num_steps=num_steps)

# if __name__ == '__init__':
