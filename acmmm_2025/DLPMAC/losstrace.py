import numpy as np
from sklearn import preprocessing
from consistent import const, adjacent_matrix
import torch
import math


def loss1(h0, h1, alpha, args):
    D0, D1 = const(h0, h0).to(args.gpu), const(h1, h1).to(args.gpu)
    W0, W1 = adjacent_matrix(h0, h0).to(args.gpu), adjacent_matrix(h1, h1).to(args.gpu)
    L0, L1 = D0 - W0, D1 - W1
    L0, L1 = L0.to(args.gpu), L1.to(args.gpu)
    L = []
    h = []
    h0, h1 = h0.type(torch.float32), h1.type(torch.float32)
    L0, L1 = L0.type(torch.float32), L1.type(torch.float32)
    h.append(h0)
    h.append(h1)
    L.append(L0)
    L.append(L1)
    R = 0
    hc = alpha[0] * h0 + alpha[1] * h1
    hc = hc.to(args.gpu)
    bdy = float(50.0)
    L0, L1 = L0.double(), L1.double()
    L0, L1 = torch.where(L0 > bdy, bdy, L0), torch.where(L1 > bdy, bdy, L1)
    for i in range(2):
        temp1 = torch.mm(hc.t(), L[i])
        temp2 = torch.mm(temp1, hc)
        temp2 = np.array(temp2.cpu())
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(temp2)
        R = R + alpha[i] * x_minmax.trace()
        alpha[i] = 1 / (2 * math.sqrt(x_minmax.trace()))
    loss = R
    alpha[1] = 1 - alpha[0]
    return loss, alpha, L0, L1
