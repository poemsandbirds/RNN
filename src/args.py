# -* coding:utf-8 *-

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training')
parser.add_argument('--num_steps', type=int, default=30,
                    help='length of sequence')
parser.add_argument('--data_path', type=str, default='data/HarryPotter/Book1.txt',
                    help='path to read data')
parser.add_argument('--num_hiddens', type=int, default=512,
                    help='number of hidden layer')
parser.add_argument('--vocab_size', type=int, default=27,
                    help='how many tokens in data')
args = parser.parse_args()
