import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
import numpy.linalg as lg
import scipy.linalg as slg
import sklearn
from sklearn import datasets

import warnings
warnings.filterwarnings('ignore')

def wasserstein_initialisation(A, B, device='cuda'):
    #Wasserstein directly on covariance
    Root_1 = slg.sqrtm(A)
    Root_2 = slg.sqrtm(B)
    C1_tilde = torch.from_numpy(Root_1.astype(np.float32)).to(device)
    C2_tilde = torch.from_numpy(Root_2.astype(np.float32)).to(device)
    return [C1_tilde, C2_tilde]

def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x + alpha * np.eye(len(x)) + np.ones([len(x), len(x)]) / len(x))
        # x_reg = torch.inverse(x + alpha * torch.eye(len(x)) + torch.ones([len(x), len(x)]) / len(x))
    else:
        x_reg = lg.pinv(x) + alpha * np.eye(len(x))
    return x_reg


