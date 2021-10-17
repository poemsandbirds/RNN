# -* coding:utf-8 *-

import numpy as np


def onehot(array, num_classes):
    num_batch = array.shape[0]
    num_labels = array.shape[1]
    index_offset = np.arange(num_labels * num_batch) * num_classes
    array_onehot = np.zeros((num_batch, num_labels, num_classes))
    # 展平的索引值对应相加，然后得到精确索引并修改array_onehot中的每一个值
    a = index_offset + array.flatten()
    b = a.astype(int)
    array_onehot.flat[b] = 1
    return array_onehot


