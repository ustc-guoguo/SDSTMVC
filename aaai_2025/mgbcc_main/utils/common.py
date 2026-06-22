import random
import numpy as np
import torch
import json
import hdf5storage as hdf
import itertools


def init_torch(seed):
    # 随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_json(path,  encoding='utf-8'):
    with open(path, "r", encoding=encoding) as fp:
        params = json.load(fp)
    return params


def load_mat(path, views=None, key_feature="data", key_label="labels"):
    dataset = path.split("/")[-1].split(".")[0]
    print(f"Loading {dataset}...")
    data = hdf.loadmat(path) # data
    num_view = 2 # len(data[key_feature]) # 视图数量
    if dataset == "BDGP" or dataset == "MNIST-USPS":
        feature = [] # 视图特征
        label = data["Y"].reshape((-1,)) # 样本标签
        # 视图特征
        feature.append(data['X1'].squeeze())
        feature.append(data['X2'].squeeze())
    elif dataset == "Hdigit":
        feature = [] # 视图特征
        label = data["truelabel"][0, 0].reshape((-1,)) # 样本标签
        # 视图特征
        feature.append(data['data'][0][1].T.squeeze())
        feature.append(data['data'][0][0].T.squeeze())
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    # 打乱样本
    # num_smp = label.size
    # rand_permute = np.random.permutation(num_smp)
    # for v in range(num_view):
    #     feature[v] = feature[v][rand_permute]
    # label = label[rand_permute]
    if views is None or len(views) == 0:
        views = list(range(num_view))
    views_feature = [feature[v] for v in views]
    return views_feature, label


# 实现功能：返回参数组合的笛卡尔集
def get_all_parameters(*parameters_range):
    return itertools.product(*parameters_range)

