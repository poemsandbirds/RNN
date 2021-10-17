# -* coding:utf-8 *-

from src.autograd import Tensor


def case_add(tensor):
    a = tensor([1., 2.], requires_grad=True)
    print(a['data'])
    b = tensor([3., 4.], requires_grad=True)
    c = tensor([[5., 6.], [7., 8.]], requires_grad=True)

    d = a + b + 10  # 相同尺寸的逐元素加法和标量加法
    # print(d)
    # d = d + c + c  # broadcast和两个操作数为同一个对象
    # print(d)
    d = d.sum()
    print(d)
    d.backward()
    print(a.grad)


def case1(tensor):
    a = tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    # b = tensor([[5., 6., 7.], [8., 9., 10.]], requires_grad=True)
    b = tensor([[0., 0., 0.], [0., 0., 0.]], requires_grad=True)
    c = a @ b
    print(c)
    c = c.sum()
    c.backward()
    print(a.grad)
# print(case_add(Tensor))
print(case1(Tensor))