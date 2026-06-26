from alignment import euclidean_dist
import torch
import numpy as np

def const(X,Y):
    #度矩阵
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    X,Y = X.to(device),Y.to(device)
    C1=euclidean_dist(X,Y)
    #C1=Hyperbole_dist(X,Y,device)
    C1=C1.to(device)
    #CD=np.where(C1<0.5,1,0)
    #CD=torch.tensor(C1)
    #CD=torch.as_tensor(torch.from_numpy(C1), dtype=torch.float32)
    D=torch.sum(C1,dim=1)
    return torch.diag(D)


def adjacent_matrix(X, Y, sigma=1):
    # """gaussian_kernel"""
    Xmul,Ymul=torch.mul(X,X),torch.mul(Y,Y)
    D2 = torch.sum(Xmul,dim=1,keepdim=True) \
          + torch.sum(Ymul,dim=1,keepdim=True).t() \
          - 2 * torch.mm(X, Y.T)
    W=torch.exp(-D2 / (2 * sigma ** 2))
    return (W+W.T)/2
    # X,Y=X.cpu(),Y.cpu()
    # X = np.array(X)
    # Y = np.array(Y)
    # D2 = np.sum(X * X, axis=1, keepdims=True) \
    #      + np.sum(Y * Y, axis=1, keepdims=True).T \
    #      - 2 * np.dot(X, Y.T)
    # print(np.shape(np.sum(X * X, axis=1, keepdims=True)),"sss")
    # W = np.exp(-D2 / (2 * sigma ** 2))
    # return (W + W.T) / 2


def cluster_matrix(y_pred):
    yt_pred = y_pred + 1
    yt_predN = 5 - y_pred
    y_len = len(y_pred)
    ndarray = []
    n_clusters = np.size(np.unique(y_pred))
    F = y_pred.reshape(y_len, 1, 1)
    Ft = yt_pred.reshape(y_len, 1, 1)
    Ft_N = yt_predN.reshape(y_len, 1, 1)
    for i in range(y_len):
        tt = np.pad(Ft[i], ((0, 0), (int(F[i]), int(Ft_N[i]))), 'constant', constant_values=(0, 0))
        ndarray.append(tt)
    ndarray = np.array(ndarray)
    FC = np.where(ndarray > 0, 1, 0)
    return FC
