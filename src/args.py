# -* coding:utf-8 *-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training')
parser.add_argument('--num_steps', type=int, default=30,
                    help='length of sequence')
parser.add_argument('--data_path', type=str, default='../data/HarryPotter/Book1.txt',
                    help='path to read data')
args = parser.parse_args()