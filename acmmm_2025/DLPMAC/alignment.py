import torch
import numpy as np


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(x, y.t(),beta=1,alpha=-2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosineSimilarty(A, B):
    A = A / (torch.norm(A, dim=1, p=2, keepdim=True) + 0.000001)
    B = B / (torch.norm(B, dim=1, p=2, keepdim=True) + 0.000001)
    W = torch.mm(A, B.t())
    return W
def cosineSimilartydis(A, B):
    A = A / (torch.norm(A, dim=1, p=2, keepdim=True) + 0.000001)
    B = B / (torch.norm(B, dim=1, p=2, keepdim=True) + 0.000001)
    W = torch.mm(A, B.t())
    return 1-W
def MD_dist(x,y):
    '''
    Manhattan distance
    Similarity matrix
    '''
    z = x.unsqueeze(1) - y.unsqueeze(0)  # p[3, 2, 4]
    z = torch.abs(z)
    pair_dist = torch.sum(z, 2, False)
    return pair_dist

