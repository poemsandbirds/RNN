# RNN
仅调用numpy实现RNN的基本功能

Quick Start:
----------------------

pip install -r requirements.txt

python predict.py

一些想法与思考
------------------------

自动求导机制的想法来自于torch.tensor，以tensor为目标进行autograd，包括梯度的存储，梯度节点的存储，反向传播等等

数据集使用的是Harry Potter系列丛书

自己用电脑训练了一下，因为不知道怎么调用gpu就用cpu跑，跑起来非常慢。但我觉得效果也算不错，跑了大概50个epoch，已经能输入harry之后写出句子了，虽然句子没有什么逻辑（可能因为训练epoch太少，数据集太少，因为我只用了哈利波特第一部）

！！！还需要完善的地方：我对比了同样是调用cpu的mxnet来训练，结果跑起来比我快了很多，可能是因为算法的结构可以优化
