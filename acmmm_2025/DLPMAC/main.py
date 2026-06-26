import torch
import random
import numpy as np
import torch.nn.functional as F
from alignment import *
# 随机生成两个 3x2 的整数矩阵
h0=torch.tensor([[5., 3.],
        [0., 7.],
        [9., 1.],
                 [3,5]])
h1=torch.tensor([[5., 4.],
        [1., 7.],
        [8., 1.],[2,5]])
print(h0)
print(h1)
def row2one(P):
    P_sum = P.sum(dim=1, keepdim=True)
    one = torch.ones(1, P.shape[1]).to(P.device)
    return P - (P_sum - 1).mm(one) / P.shape[1]

def col2one(P):
    P_sum = P.sum(dim=0, keepdim=True)
    one = torch.ones(P.shape[0], 1).to(P.device)
    return P - (one).mm(P_sum - 1) / P.shape[0]

def P_init(D):
    P = torch.zeros_like(D)
    D_rowmin = D.clone()
    max_d = D.max()
    min_ind = torch.argmin(D_rowmin, dim=0)
    D_rowmin[:, :] = max_d
    D_rowmin = D_rowmin.scatter(0, min_ind.unsqueeze(0), D[min_ind, torch.arange(D.shape[1]).long()].unsqueeze(0))

    _, idx_max = torch.min(D_rowmin, dim=1)

    P[torch.arange(D.shape[0]).long(), idx_max.long()] = 1.0

    return P
def alignment(D, tau_1=30, tau_2=10, lr=0.1):
    P = P_init(D)

    d = [torch.zeros_like(D) for _ in range(3)]

    for i in range(tau_1):
        P = P - lr * D
        for j in range(tau_2):
            P_0 = P.clone()
            P = P + d[0]
            Y = row2one(P)
            d[0] = P - Y

            P = Y + d[1]
            Y = col2one(P)
            d[1] = P - Y

            P = Y + d[2]
            Y = F.relu(P)
            d[2] = P - Y

            P = Y
            if (P - P_0).norm().item() == 0:
                break

    return P
n_samples = h0.shape[0]
P_index = random.sample(range(n_samples), n_samples)
P_gt = np.eye(n_samples).astype('float32')
P_gt = P_gt[:, P_index]
P_gt=torch.from_numpy(P_gt)
h1=torch.mm(P_gt,h1)
print(P_gt)
print(h1)
C = euclidean_dist(h0, h1)
print(C)
P_pred = alignment(C)
print(P_pred)
print(P_gt)
print(torch.abs(P_gt-P_pred))
print(torch.abs(P_gt.T-P_pred))
print(torch.sum(torch.abs(P_gt-P_pred)))
print(torch.sum(torch.abs(P_gt.T-P_pred)))
