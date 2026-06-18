
import random
import numpy as np
import torch
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from utils import min_max_normalize, euclidean_distance_np, euclidean_distance
from datasets import myDataset

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True




# # ================================== euclidean_distance_np ==================
# data1 = np.random.randn(5, 4)
# data2 = np.random.randn(6, 4)
# print(f"data1: \n{data1}")
# print(f"data2: \n{data2}")
# euc_dis_np = euclidean_distance_np(data1, data2)
# print(f"euc_dis_np: {euc_dis_np.shape}\n{euc_dis_np}")
# data1_ = torch.from_numpy(data1).float()
# data2_ = torch.from_numpy(data2).float()
# euc_dis_tch = euclidean_distance(data1_, data2_, device=torch.device('cpu'))
# print(f"euc_dis_tch: {euc_dis_tch.shape}\n{euc_dis_tch}")
# # ===========================================================================

# # ================================== realigne data =======================
# datasets = {
#         0: "HandWritten", 1: "Scene-15", 2: "BDGP", 3: "Caltech101-7", 4: "Caltech101-20",
#         5: "Reuters_dim10", 6: "MNIST-USPS", 7: "NoisyMNIST", 8: "CUB", 9: "Wiki",
#         10: "Hdigit", 11: "NUS-WIDE", 12: "Deep Animal", 13:"Fashion"
#     }
# data_id = 0
# data_name = datasets[data_id]
# aligned_ratio = 0.5
# data_root = "D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA"
# mydata = myDataset(data_name=data_name,
#                    root=data_root,
#                    aligned_ratio=aligned_ratio)
# x_view1, x_view2, label1, label2 = mydata.get_partially_aligned_data()
# x1_u, x2_u = x_view1[~mydata.flag], x_view2[~mydata.flag]
# label1_u, label2_u = label1[~mydata.flag], label2[~mydata.flag]
# dim1, dim2 = x_view1.shape[1], x_view2.shape[1]
# if dim1 != dim2:
#     dim_ = min(dim1, dim2)
#     pca = PCA(n_components=dim_)
#     if dim1 < dim2:
#         x2_u = pca.fit_transform(x2_u)
#     else:
#         x1_u = pca.fit_transform(x1_u)
# dis = euclidean_distance_np(x1_u, x2_u)
# row_indices, col_indices = linear_sum_assignment(dis)
# x2_realigned = x_view2.copy()
# label2_realigned = label2.copy()
# label2_u2 = label2_u
# x2_realigned[~mydata.flag] = x2_realigned[~mydata.flag][col_indices]
# label2_u2 = label2_u2[col_indices]
# label2_realigned[~mydata.flag] = label2_realigned[~mydata.flag][col_indices]
# cnt1, cnt2 = 0, 0
# for i in range(len(label1)):
#     if label1[i] == label2[i]:
#         cnt1 += 1
#     if label1[i] == label2_realigned[i]:
#         cnt2 += 1
# print(f"num samples: {len(label1)}, label2_realigned: {label2_realigned.shape}, cnt1: {cnt1}, cnt2: {cnt2}")
# print(f"x_view1: {x_view1.shape}, x_view2: {x_view2.shape}, x2_realigned: {x2_realigned.shape}")
# # num samples: 2000, label2_realigned: (2000,), cnt1: 1106, cnt2: 1441
# # x_view1: (2000, 240), x_view2: (2000, 216), x2_realigned: (2000, 216)
# realigned_data = {"X1": x_view1, "X2": x2_realigned, "Y": label1}
# save_path = f"D:/LPeng/AI/Datasets/MVC_datasets/RealignedData/{data_name}_{aligned_ratio}.mat"
# # sio.savemat(save_path, realigned_data)
# # ===========================================================================


# # ================================== min_max_normalize ======================
# # data1 = torch.randint(0, 10, (5, 5))
# data1 = torch.tensor([[7, 5, 2, 7, 2],
#                       [0, 0, 0, 0, 0],
#                       [6, 3, 1, 0, 6],
#                       [3, 4, 0, 6, 2],
#                       [8, 9, 2, 0, 9]])
# data2 = min_max_normalize(data1)
# print(f"data1: \n{data1}")
# print(f"data2: \n{data2}")
# max_ = torch.max(data1)
# min_ = torch.min(data1)
# print(f"max_: {max_}")
# print(f"min_: {min_}")
# data3 = (data1 - min_) / (max_ - min_)
# print(f"data3: \n{data3}")
# # ===========================================================================

# # ================================== aligned_ratio ==========================
# aligned_ratio = 0.5
# label, label2 = [], []
# numbers = [10, 50, 30, 5, 5]
# for y in range(len(numbers)):
#     for n in range(numbers[y]):
#         label.append(y)
# print(f"label: {len(label)}, {len(set(label))}, {set(label)}\n{label}")
# n_aligned_dict, index_all_dict, flag_dict = {}, {}, {}
# flag = np.zeros(len(label)).astype(np.bool_)
# print(f"flag: {len(flag)}, {flag}")
# for y in np.unique(label):
#     indices_y = np.where(label == y)[0]
#     index_all_dict[y] = indices_y
#     # print(f"indices_y: {indices_y}")
#     n_aligned = int(np.ceil(len(indices_y) * aligned_ratio))
#     n_aligned_dict[y] = n_aligned
#     indices_y_ = shuffle(indices_y)
#     flag[indices_y_[:n_aligned]] = True
# print(f"index_dict: {index_all_dict}")
# print(f"n_aligned_dict: {n_aligned_dict}")
# print(f"flag: {len(flag)}, {flag}")
# # ===========================================================================

# ================================== WIKI Hdigit ============================
datasets = ['WIKI', 'Hdigit', 'NUS-WIDE', 'OutScene', 'ORL',
             'prokaryotic', 'Pascal', 'RGB-D', 'Uci-digit',
             'LandUse-21', 'Fashion', 'Flowers17', 'Yale',
             'caltech20', 'Caltech101-all', 'Deep Animal', 'YoutubeFace_sel_fea']
data_name = 'YoutubeFace_sel_fea'
data_path = "D:/LPeng/AI/Datasets/MVC_datasets"
# data_path = "D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA"
data = sio.loadmat(f"{data_path}/{data_name}.mat")
print(f"data: {data}")
label = np.squeeze(data['Y'])
print(f"label: {label.shape}")      # label: (1, 2866)
print(f"num classes: {len(np.unique(label))}")
data1 = data['X'][0, 0]
print(f"data1: {data1.shape}")     # Img: (2866, 128)
data2 = data['X'][1, 0]
print(f"data2: {data2.shape}")     # Txt: (2866, 10)
data3 = data['X'][2, 0]
print(f"data3: {data3.shape}")     # Img: (2866, 128)
data4 = data['X'][3, 0]
print(f"data4: {data4.shape}")     # Txt: (2866, 10)
data5 = data['X'][4, 0]
print(f"data5: {data5.shape}")     # Img: (2866, 128)
# data6 = data['X'][0, 5].T
# print(f"data6: {data6.shape}")     # Txt: (2866, 10)
# data7 = data['X'][0, 6].T
# print(f"data7: {data7.shape}")     # Txt: (2866, 10)
# X_list = []
# X = data['data'][0]
# X_list.append(X[0].T.astype('float32'))
# X_list.append(X[1].T.astype('float32'))
# label0 = np.squeeze(data['truelabel'][0, 0])
# label1 = np.squeeze(data['truelabel'][0, 1])
# print(f"X_list: {X_list}")
# print(f"label0: {label0}")
# flag = label0 == label1
# print(f"flag: {flag.astype(np.int64()).sum()}")
# ===========================================================================

# # ================================== not all zeros ==========================
# # data1 = torch.randint(0, 5, (10, 10))
# data1 = torch.tensor([[-2, 0, 2, 2, -2, 0, 2, -2, 1, 0],
#         [1, -3, 1, 0, -1, 3, 4, 0, -1, 2],
#         [3, -4, 2, 0, 4, -4, 4, 4, -4, 4],
#         [4, 0, -4, 0, 3, -4, 3, 0, 4, -3],
#         [2, -4, 0, 2, -4, 1, 1, -2, 3, 4],
#         [0, -1, -5, 0, -1, 0, -3, 0, -2, 0],
#         [-4, 4, 4, -4, 4, 4, -3, 3, 1, -3],
#         [2, 1, -1, 4, -4, 1, 3, 4, -2, 4],
#         [-3, 0, -2, -1, 0, 0, -2, 0, -4, -1],
#         [0, -2, 1, 3, -2, 2, -4, 0, 0, 2]], dtype=torch.float32)
# data2 = torch.where(data1 > 0, data1, 0)
# print(f"data2: {data2}")
# sum_row_ = torch.sum(data2, dim=-1)
# print(f"sum_row_: {sum_row_}")
# indices_zero_ = torch.where(sum_row_ == 0)
# print(f"indices_zero_: {len(indices_zero_[0])}, {indices_zero_}, {indices_zero_[0]}")
# print(f"data1[indices_zero_[0]]: {data1[indices_zero_[0]]}")
# if len(indices_zero_[0]) > 0:
#     idices_max_ = torch.argmax(data1[indices_zero_[0]], dim=-1)
#     print(f"idices_max_: {idices_max_}")
#     data2[indices_zero_[0], idices_max_] = 0.01
#     print(f"data2: {data2}\n{torch.sum(data2, dim=-1)}")
# # ===========================================================================

# # ================================== data1 / data2 div() ====================
# data1 = torch.tensor([10, 15, 16, 20, 22, 28, 30, 35, 36])
# data2 = torch.tensor([2, 5, 8, 4, 11, 7, 3, 5, 6])
# print(f"data1/data2: {data1 / data2}")
# print(f"data1.div(): {data1.div(data2)}")
# print(f"torch.div(): {torch.div(data1, data2)}")
# data3 = torch.diag(data1)
# print(f"data3: {data3}")
# print(f"data3: {torch.diag(data3)}")
# # t = -0.0001
# # print(max((t, 0)))
# # ===========================================================================

# # ================================== cov_matrix ===================================
# data = torch.randn((10, 6))
# N, D = data.size()
# print(f"data: {data}")
# mean_ = torch.mean(data, dim=-1).unsqueeze(-1)
# std_ = torch.std(data, dim=-1).unsqueeze(-1)
# ones_row_vec = torch.ones(data.size(-1), dtype=torch.float32).unsqueeze(0)
# print(f"mean: \n{mean_}")
# print(f"std: \n{std_}")
# print(f"ones_col_vec: \n{ones_row_vec}")
# data = (data - (torch.mm(mean_, ones_row_vec))).div(torch.mm(std_, ones_row_vec))
# print(f"mean_: {torch.mm(mean_, ones_row_vec)}")
# print(f"std_: {torch.mm(std_, ones_row_vec)}")
# print(f"data: {data}\nmean: {torch.mean(data, dim=-1)}, std: {torch.std(data, dim=-1)}")
# cov_matrix = torch.mm(data, data.t()) / (D-1)
# print(f"cov_matrix: \n{cov_matrix}")
# # temp = (torch.std(data, dim=-1) * torch.std(data, dim=-1)).unsqueeze(-1)
# # temp = torch.mm(temp, torch.ones(cov_matrix.size(-1), dtype=torch.float32).unsqueeze(0))
# # cov_matrix = cov_matrix.div(torch.mm(std_, std_.t()))
# # print(f"temp: \n{torch.mm(std_, std_.t())}")
# # print(f"cov_matrix: \n{cov_matrix}")
# # ===========================================================================

# # ================================== BDGP ===================================
# data_path = "D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA"
# data = sio.loadmat(f"{data_path}/BDGP_raw.mat")
# print(f"data: {data}")
# data1 = {}
# data1['X1'] = data['X2']
# data1['X2'] = data['X1']
# data1['Y'] = data['Y']
# print(f"data1: {data1}")
# data_path = f"{data_path}/BDGP.mat"
# sio.savemat(data_path, data1)
# # ===========================================================================

# # ================================== torch.diag mask ========================
# data = torch.randint(10, (10, 10))
# print(f"data: \n{data}")
# ones = torch.ones(10).bool()
# print(f"ones: \n{ones}")
# data1 = data.clone()
# flag = torch.randint(2, (10,)).bool()
# print(f"flag: \n{flag}")
# n_aligned = torch.sum(flag.int())
# print(f"n_aligned: {n_aligned}")
# data1[flag, flag] = 1
# print(f"data1: \n{data1}")
# data2 = data1[flag, :]
# data2 = data2[:, flag]
# print(f"data2: \n{data2}")
# data3 = data1[flag, flag]
# print(f"data3: \n{data3}")
# print(f"data3diag: \n{torch.diag(data3)}")
# idx = 5
# diag1 = torch.diag(data, idx)
# print(f"diag1: {diag1}")
# diag2 = torch.diag(data, -idx)
# print(f"diag2: {diag2}")
# mask = torch.eye(data.size()[0]).bool()
# idx2 = int(data.size(0)/2)
# for i in range(idx2):
#     mask[i, idx2+i] = True
#     mask[idx2+i, i] = True
# print(f"mask: \n{mask}")
# data[mask] = 0
# print(data)
# data4 = data[mask].reshape(10, -1)
# print(f"data4: \n{data4}")
# # ===========================================================================

# # ================================== torch.where ============================
# data = torch.randn((5, 5))
# print(f"data: {data}")
# data = data - torch.diag_embed(torch.diag(data))
# print(f"data: {data}")
# data = torch.where(data > 0, data, 0)
# print(f"data: {data}")
# # ===========================================================================

# # ================================== mean std ===================================
# data = torch.randn(20)
# print(f"data: {len(data)}, {data}")
# mean_ = torch.mean(data)
# std_ = torch.std(data)
# min_ = torch.min(data)
# print(f"mean: {mean_}")
# print(f"std: {std_}")
# print(f"min: {min_}")
# print(f"mean - std: {mean_ - std_}")
# weights = torch.zeros(data.size())
# indices_neu = data > mean_ - std_
# print(f"indices_neu: {indices_neu}")
# weights[indices_neu] = data[indices_neu]
# print(f"weights: {weights}")
# flag = torch.tensor([True, True, True, False, True, True, True, True, True, True,
#                      True, True, True, True, True, False, True, False, False, True])
# indices = data > 0
# print(f"flag: {flag}")
# print(f"indices: {indices}")
# weights[indices] = 1
# print(f"weights: {weights}")
# weights[flag] = 1
# print(f"weights: {weights}")
# # ===========================================================================

# # ================================== aligned_idx ============================
# num_samples, aligned_rate = 20, 1.0
# split_idx = np.random.permutation(num_samples)
# print(f"split_idx: {len(split_idx)}, {split_idx}")
# print(np.sort(split_idx))
# aligned_num = int(np.ceil(aligned_rate * num_samples))
# print(f"aligned_num: {aligned_num}")
# aligned_idx = split_idx[:aligned_num]
# unaligned_idx = split_idx[aligned_num:]
# print(f"aligned_idx: {len(aligned_idx)} {aligned_idx}")
# print(f"unaligned_idx: {len(unaligned_idx)} {unaligned_idx}")
# # ===========================================================================

# # ================================== flag ===================================
# num_sample = 20
# P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
# print(f"P_index: {P_index}")        # P_index: [ 0  1  2 ... 18 19]
# flag = np.random.randint(0, 2, 20)
# print(f"flag: {flag.shape} {flag}") # flag: (20,) [1 1 0 1 0 ... 0 1 0 0 0 0]
# flag = flag > 0
# print(f"flag: {flag.shape} {flag}") # flag: (20,) [ True True False True False ... False False]
# sub_index0 = P_index[~flag]
# sub_index1 = P_index[flag]
# print(f"sub_index0: {sub_index0.shape} {sub_index0}")   # sub_index0: (10,) [ 2  4  7 10 13 14 16 17 18 19]
# print(f"sub_index1: {sub_index1.shape} {sub_index1}")   # sub_index1: (10,) [ 0  1  3  5  6  8  9 11 12 15]
# # ===========================================================================

# dim = 2000
# print(dim ** (-0.5))

# def euclidean_dist(x, y):
#     """
#     From https://github.com/XLearning-SCU/2021-CVPR-MvCLN
#     Args:
#         x: pytorch Variable, with shape [m, d]
#         y: pytorch Variable, with shape [n, d]
#     Returns:
#         dist: pytorch Variable, with shape [m, n]
#     """
#
#     m, n = x.size(0), y.size(0)
#     # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     # yy会在最后进行转置的操作
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
#     dist.addmm_(1, -2, x, y.t())
#     # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     return dist

# # ================================ KNN =====================================
# def compute_knn(sim_matrix, k=3, device=torch.device('cpu')):
#     # diag = torch.diag(sim_matrix)
#     # sim_matrix = sim_matrix - torch.diag_embed(diag)
#     values, indices = torch.topk(sim_matrix, k, dim=-1, largest=True)
#     adj = torch.zeros(sim_matrix.size(), dtype=torch.int64, device=device)
#     # adj[indices] = 1
#     adj = adj.scatter_(1, indices, 1)  # 在每行中按照列索引将值 scatter 到目标张量中
#     # adj = adj + adj.t()
#     # adj = torch.clamp(adj, max=1)
#     return adj
# k=3
# score = torch.randint(0, 10, (10, 10))
# print(f"score: {score.size()}\n{score}")
# adj = compute_knn(score, k)
# print(f"adj: \n{adj}")
# # ==========================================================================

# # ================================== scatter_ =============================
# P = torch.tensor([[0.1, 0, 0.3, 0],
#                 [0, 0.1, 0.5, 0],
#                 [0, 0.2, 0, 0.1],
#                 [0.2, 0, 0, 0.4]])
# max_val, max_idx = torch.max(P, dim=1, keepdim=True)
# P_pred = torch.zeros_like(P)
# P_pred.scatter_(1, max_idx, 1)
# print(f"P_pred: {P_pred}")
# # P_pred: tensor([[0., 0., 1., 0.],
# #                 [0., 0., 1., 0.],
# #                 [0., 1., 0., 0.],
# #                 [0., 0., 0., 1.]])
# print(f"PP^T: {torch.matmul(P_pred, P_pred.t())}")
# ones = torch.ones(P.shape[0]).view(P.shape[0], 1)
# print(f"1: {ones}")
# print(f"P1: {torch.matmul(P_pred, ones)}")
# print(f"P^T1: {torch.matmul(P_pred.t(), ones)}")
# # =========================================================================

# # ================================== diag =================================
# adj = torch.tensor([[0.1, 0, 0.3, 0],
#                     [0, 0.1, 0.5, 0],
#                     [0, 0.1, 0, 0.1],
#                     [0.2, 0, 0, 0.4]])
# diag_adj = torch.diag(adj)
# print(f"diag_adj: {diag_adj}")              # tensor([0.1000, 0.1000, 0.0000, 0.4000])
# print(f"diag_adj*: {diag_adj*diag_adj}")    # tensor([0.0100, 0.0100, 0.0000, 0.1600])
# print(f"diag_embed: {torch.diag_embed(diag_adj)}")
# print(f"adj - diag: {adj - torch.diag_embed(diag_adj)}")
# # =========================================================================

# # ================================ omega ==================================
# adj = torch.tensor([[0.1, 0, 0.3, 0],
#                     [0, 0.1, 0.5, 0],
#                     [0, 0.1, 0, 0.1],
#                     [0.2, 0, 0, 0.4]])
# omega = (adj > 0).int()
# omega = torch.sum(adj, dim=-1) / torch.sum(omega, dim=-1)
# print(f"omega: {omega}")
# omega1 = torch.unsqueeze(omega, dim=-1)
# print(f"omega1: {omega1}")
# print(f"weighted: {torch.mul(omega1, adj)}")
# print(omega**2)
# # =========================================================================

# # ================================ shuffle_index ==========================
# def shuffle_index(num_sample, aligned_ratio=0.0, random_state=2):
#     """
#     根据样本数量和对齐率进行分配
#         inputs:
#             num_sample: 样本数量
#             aligned_ratio: 对齐率
#         returns:
#             flag: 长度为n的分配列表，其中 True: 表示该索引位置两个视图的数据是对齐的，False: 表示该索引位置两个视图的数据不是对齐的
#     """
#     num_aligned = int(num_sample * aligned_ratio)
#     # 打乱索引
#     index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
#     index = shuffle(index, random_state=random_state)
#     # 分配, 取打乱后的前一部分索引位置作为对齐数据
#     flag = index < 0
#     flag[index[:num_aligned, ]] = True
#
#     P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
#     # index_mis_aligned = shuffle(P_index, random_state=2)
#     index_mis_aligned = shuffle(P_index[~flag], random_state=2)
#     P_index[~flag] = index_mis_aligned
#
#     return P_index, flag, index_mis_aligned
#
# num_sample = 10
# index_shuffled, flag, index_mis_aligned = shuffle_index(num_sample, aligned_ratio=0.5)
# print(f"index_shuffled: {index_shuffled.shape}, {index_shuffled}")              # index_shuffled: (10,), [0 1 6 9 4 5 3 7 8 2]
# print(f"flag: {flag.shape}, {flag}")                                            # flag: (10,), [ True  True False False  True  True False  True False False]
# print(f"index_mis_aligned: {index_mis_aligned.shape}, {index_mis_aligned}")     # index_mis_aligned: (5,), [6 9 3 8 2]
# # =========================================================================================


# # ============================= dataset info ==============================================
# datasets = {
#         0: "HandWritten", 1: "Scene-15", 2: "BDGP", 3: "Caltech101-7", 4: "Caltech101-20",
#         5: "Reuters_dim10", 6: "MNIST-USPS", 7: "NoisyMNIST", 8: "CUB", 9: "3sources",
#         10: "BBC4", 11: "Deep Animal"
#     }
# data_id = 3
# data_path = f"D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA/{datasets[data_id]}.mat"
# mat = sio.loadmat(data_path)
# print(mat)
# # data = mat
# # data = {}
# # data["X"] = mat["x_test"]
# # # data["X2"] = mat["X2"]
# # # # # # # # data["X3"] = mat["data"][0][2]
# # data["Y"] = mat["y_test"]
# # n_samples = data['X'].shape
# # dim1 = data['X'][0][0].shape[1]
# # dim2 = data['X'][0][1].shape[1]
# # dim3 = data['X'][0][2].shape[1]
# # dim4 = data['X'][0][3].shape[1]
# # dim5 = data['X'][0][4].shape[1]
# # dim6 = data['X'][0][5].shape[1]
# # # # # dim7 = data['X'][0][6].shape[1]
# # # # # # # n_classes = 0
# # n_classes = len(np.unique(np.squeeze(data['Y'])))
# # print(f"{datasets[data_id]} {n_samples} {n_classes} {dim1} {dim2} {dim3} {dim4}")     # {dim1} {dim2} {dim3} {dim4} {dim5} {dim6}
# # print(data)
# # print(f"n_samples: {data['X1'].shape[0]}")                      # n_samples: 600
# # print(f"n_classes: {len(np.unique(np.squeeze(data['Y'])))}")    # n_classes: 10
# # sio.savemat(f'D:/LPeng/AI/Datasets/MVC_datasets/{datasets[data_id]}_new.mat', data)
# # data = sio.loadmat(f'D:/LPeng/AI/Datasets/MVC_datasets/{datasets[data_id]}_new.mat')
# # print(data)
# # =======================================================================================


# # ============================= torch.mul() ================================
# tensor0 = torch.tensor([[1, 2, 3, 4],
#                         [6, 5, 4, 3],
#                         [3, 5, 2, 6],
#                         [5, 1, 8, 3]])
# adj = torch.tensor([[1, 0, 1, 0],
#                     [0, 1, 0, 0],
#                     [0, 1, 0, 1],
#                     [1, 0, 0, 1]])
# tensor1 = tensor0 * adj
# tensor2 = torch.mul(tensor0, adj)
# tensor3 = torch.sum(tensor0, dim=-1)
# tensor4 = torch.diag(tensor0) + tensor3
# print(f"tensor1: \n{tensor1}")      # tensor([[1, 0, 3, 0], ... [5, 0, 0, 3]])
# print(f"tensor2: \n{tensor2}")      # tensor([[1, 0, 3, 0], ... [5, 0, 0, 3]])
# print(f"diag: {torch.diag(tensor0).size()}\n{torch.diag(tensor0)}")     # diag: torch.Size([4]) tensor([1, 5, 2, 3])
# print(f"sum: {tensor3.size()}\n{tensor3}")      # sum: torch.Size([4])  tensor([10, 18, 16, 17])
# print(f"add: {tensor4.size()}\n{tensor4}")      # add: torch.Size([4])  tensor([11, 23, 18, 20])
# # ===========================================================================

# # ================================ torch.concat() ===========================
# tensor0 = torch.tensor([[1, 2, 3, 4],
#                         [6, 5, 4, 3],
#                         [3, 5, 2, 6],
#                         [5, 1, 8, 3]])
# tensor1 = tensor0.t()
# tensor2 = torch.concat((tensor0, tensor1), dim=-1)
# tensor3 = torch.concat((tensor0, tensor1), dim=0)
# print(f"tensor1: {tensor1.size()}\n{tensor1}")      # tensor1: torch.Size([4, 4])
# print(f"tensor2: {tensor2.size()}\n{tensor2}")      # tensor2: torch.Size([4, 8])
# print(f"tensor3: {tensor3.size()}\n{tensor3}")      # tensor3: torch.Size([8, 4])
# # ===========================================================================

# # =========================== set max to 1, others to 0 ===================
# x = torch.randn([10, 10]).cuda()
# mask = (x == x.max(dim=-1, keepdim=True)[0]).to(dtype=torch.int32)
# print(f"x: \n{x}")
# print(f"mask: \n{mask}")
# # =========================================================================

# # # ==================== torch.diag =====================================
# tensor0 = torch.tensor([[1, 2, 3, 4],
#                         [6, 5, 4, 3],
#                         [3, 5, 2, 6],
#                         [5, 1, 8, 3]]).float()
# diag = torch.diag(tensor0)
# tensor1 = tensor0 - torch.diag_embed(diag)
# tensor2 = torch.diag_embed(diag)
# print(f"diag: {diag.size()}\n{diag}")               # diag: torch.Size([4])
# print(f"tensor2: {tensor2.size()}\n{tensor2}")      # tensor2: torch.Size([4, 4])
# print(f"tensor1: {tensor1.size()}\n{tensor1}")      # tensor1: torch.Size([4, 4])
# # # =======================================================================

# # # ==================== torch.norm =====================================
# tensor0 = torch.tensor([[1, 2, 3, 4, 5, 6],
#                         [6, 5, 4, 3, 2, 1],
#                         [3, 5, 2, 6, 4, 1]]).float()
# tensor1 = tensor0 / torch.norm(tensor0, dim=-1, keepdim=True)
# tensor2 = torch.matmul(tensor0, tensor1.t())
# print(f"tensor1: {tensor1}")
# print(f"tensor2: {tensor2}")
# print(f"tensor3: {(tensor1 * tensor1)}")
# print(f"tensor3 sum: {(tensor1 * tensor1).sum(dim=-1)}")        # tensor3 sum: tensor([1.0000, 1.0000, 1.0000])
# # # ======================================================================

# # ==================================================================================
# def pseudo_graph(label_pred):
#     label_pred = torch.tensor(label_pred)
#     pseudo_g = (label_pred == label_pred.unsqueeze(1)).float()
#     diag = torch.diag(pseudo_g)
#     pseudo_g = pseudo_g - torch.diag_embed(diag)
#
#     return pseudo_g

# label0 = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
# adj0 = torch.matmul(label0.unsqueeze(-1), label0.unsqueeze(0))
# print(f"adj0: {adj0}")
#
# label1 = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
# label2 = torch.tensor([1, 2, 1, 2, 1, 3, 3, 2, 3])
# adj1 = torch.matmul(label1.unsqueeze(-1), label2.unsqueeze(0))
# print(f"adj1: {adj1}")
# p_label1 = torch.tensor([1, 0, 1, 0, 2, 2, 0, 3, 3])
# p_label2 = torch.tensor([1, 0, 2, 2, 1, 0, 3, 2, 3])
# adj2 = torch.matmul(p_label1.unsqueeze(-1), p_label2.unsqueeze(0))
# print(f"adj2: {adj2}")
# # ==================================================================================


# # ========================= random.sample(range(m), n) ===========================
# random_idx = random.sample(range(10), 10)   # randomly select n numbers from list range(10)
# print(f"random_idx: {random_idx}")          # random_idx: [9, 0, 6, 3, 4, 8, 1, 7, 2, 5]  5: [9, 0, 6, 3, 4]
# # ================================================================================


# # # =================== torch.where(tensor >= 3) =================================
# arr = np.array([1, 2, 3, 4, 5, 6])
# ind_arr = np.squeeze(np.argwhere(arr >= 3), axis=-1)
# print(f"ind_arr: {ind_arr}")                            # ind_arr: [2 3 4 5]
# print(f"a[ind_arr]: {arr[ind_arr]}")                    # a[ind_arr]: [3 4 5 6]
# arr[ind_arr] = -1
# print(f"arr: {arr}")                                    # arr: [ 1  2 -1 -1 -1 -1]
#
# tensor = torch.tensor([1, 2, 3, 4, 5, 6])
# ind_tensor = torch.where(tensor >= 3)[0]
# print(f"ind_tensor: {ind_tensor}")                   # ind_tensor: tensor([2, 3, 4, 5])
# print(f"t[ind_tensor]: {tensor[ind_tensor]}")           # t[ind_tensor]: tensor([3, 4, 5, 6])
# tensor[ind_tensor] = -1
# print(f"tensor: {tensor}")                              # tensor: tensor([ 1,  2, -1, -1, -1, -1])
# # # ==============================================================================


# # ==================== scatter_(1, indices, 1) =================================
# tensor0 = torch.tensor([[1, 2, 3, 4, 5, 6],
#                         [6, 5, 4, 3, 2, 1],
#                         [3, 5, 2, 6, 4, 1]])
# values, indices = torch.topk(tensor0, 2, dim=-1)  # top k neighbor
# pseudo_adj = torch.zeros_like(tensor0)
# # 在每行中按照列索引将值 scatter 到目标张量中。dim=1: indices为每行中的列索引；dim=0: indices为每列中的行索引
# pseudo_adj = pseudo_adj.scatter_(1, indices, 1)
# print(f"pseudo_adj: {pseudo_adj}")            # tensor([[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0]])
# # ==============================================================================


# # ========================= [a] + list1 + list2 ==================================
# tensor0 = torch.tensor([1, 2, 3])
# list1 = []
# list1.append(torch.tensor([1, 2, 3, 4]))
# list2 = []
# list2.append(torch.tensor([1, 2, 3, 4, 5, 6]))
# list3 = []
# list4 = [tensor0] + list1 + list2 + list3
# print(f"list4: {list4}")            # list3: [tensor([1, 2, 3]), tensor([1, 2, 3, 4]), tensor([1, 2, 3, 4, 5, 6])]
# # ================================================================================


# # ======================== trAB ===========================================
# A = torch.randn((5, 4))
# B = torch.randn((4, 5))
# print(f"trAB: {torch.trace(torch.mm(A, B))}")       # trAB: -2.5129318237304688
# print(f"trBA: {torch.trace(torch.mm(B, A))}")       # trBA: -2.5129313468933105
# # =========================================================================

