
import random
import numpy as np
import torch
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from utils import euclidean_distance_np, euclidean_distance
from datasets import myDataset

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

datasets = {
        0: "HandWritten", 1: "Scene-15", 2: "BDGP", 3: "Caltech101-7", 4: "Caltech101-20",
        5: "Reuters_dim10", 6: "MNIST-USPS", 7: "NoisyMNIST", 8: "CUB", 9: "Wiki",
        10: "Hdigit", 11: "NUS-WIDE", 12: "Deep Animal", 13:"Fashion"
    }
data_id = 9
data_name = datasets[data_id]
aligned_ratio = 0.5
lack_memory = True
data_root = "D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA"
mydata = myDataset(data_name=data_name,
                   root=data_root,
                   aligned_ratio=aligned_ratio)
x_view1, x_view2, label1, label2 = mydata.get_partially_aligned_data()
x1_u, x2_u = x_view1[~mydata.flag], x_view2[~mydata.flag]
label1_u, label2_u = label1[~mydata.flag], label2[~mydata.flag]
print(f"All: {len(x_view1)}, aligned: {len(label1[mydata.flag])}, unaligned: {len(label1_u)}")
dim1, dim2 = x_view1.shape[1], x_view2.shape[1]
if dim1 != dim2:
    dim_ = min(dim1, dim2)
    if dim_ > len(label1_u):
        dim_ = len(label1_u)
    print(f"PCA n_components: {dim_}")
    pca = PCA(n_components=dim_)
    if dim1 < dim2 and dim_ == min(dim1, dim2):
        x2_u = pca.fit_transform(x2_u)
    elif dim1 > dim2 and dim_ == min(dim1, dim2):
        x1_u = pca.fit_transform(x1_u)
    elif dim_ < min(dim1, dim2):
        x1_u = pca.fit_transform(x1_u)
        x2_u = pca.fit_transform(x2_u)
dis = euclidean_distance_np(x1_u, x2_u, lack_memory=lack_memory)
row_indices, col_indices = linear_sum_assignment(dis)
x2_realigned = x_view2.copy()
label2_realigned = label2.copy()
label2_u2 = label2_u
x2_realigned[~mydata.flag] = x2_realigned[~mydata.flag][col_indices]
label2_u2 = label2_u2[col_indices]
label2_realigned[~mydata.flag] = label2_realigned[~mydata.flag][col_indices]
cnt1, cnt2 = 0, 0
for i in range(len(label1)):
    if label1[i] == label2[i]:
        cnt1 += 1
    if label1[i] == label2_realigned[i]:
        cnt2 += 1
print(f"num samples: {len(label1)}, label2_realigned: {label2_realigned.shape}, cnt1: {cnt1}, cnt2: {cnt2}")
print(f"x_view1: {x_view1.shape}, x_view2: {x_view2.shape}, x2_realigned: {x2_realigned.shape}")
# num samples: 2000, label2_realigned: (2000,), cnt1: 1106, cnt2: 1441
# x_view1: (2000, 240), x_view2: (2000, 216), x2_realigned: (2000, 216)
realigned_data = {"X1": x_view1, "X2": x2_realigned, "Y": label1}
print(f"realigned_data: {realigned_data}")
save_path = f"D:/LPeng/AI/Datasets/MVC_datasets/RealignedData/{data_name}_{aligned_ratio}.mat"
sio.savemat(save_path, realigned_data)

data = sio.loadmat(save_path)
print(f"data: {data}")

# # ================================== euclidean_distance_np ==================
# data1 = np.random.randn(5, 4)
# data2 = np.random.randn(6, 4)
# data3 = np.random.randn(7, 10)
# print(f"data1: \n{data1}")
# print(f"data2: \n{data2}")
# euc_dis_np = euclidean_distance_np(data1, data2)
# print(f"euc_dis_np: {euc_dis_np.shape}\n{euc_dis_np}")
# euc_dis_np2 = euclidean_distance_np(data1, data2, lack_memory=True)
# print(f"euc_dis_np2: {euc_dis_np2.shape}\n{euc_dis_np2}")
# data1_ = torch.from_numpy(data1).float()
# data2_ = torch.from_numpy(data2).float()
# euc_dis_tch = euclidean_distance(data1_, data2_, device=torch.device('cpu'))
# print(f"euc_dis_tch: {euc_dis_tch.shape}\n{euc_dis_tch}")
# euc_dis_np3 = euclidean_distance_np(data1, data3)
# # ===========================================================================
