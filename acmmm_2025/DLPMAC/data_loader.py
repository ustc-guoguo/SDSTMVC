import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import TT_split, normalize
import torch
import random
import mat73
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint
def load_data(align_prop,complete_prop,neg_num,is_noise,dataset):
    all_data = []
    train_pairs = []
    label = []
    traindata=[]
    if dataset=='Caltech101_7':
        path = './datasets/' + dataset + '.mat'  # 路径
        mat = mat73.loadmat(path)  # 加载mat文件
    else:
        mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'HandWritten':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == '3Sources':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'MSRCv1':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == 'BBCsports':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'NoisyMNIST_10000':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'BBC4':
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == 'BDGP':
        # mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'ORL_mtv':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'Caltech101_7':
        data = mat['data'][3:5]
        label = np.squeeze(mat['labels'])
    elif dataset == 'flower17':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Prokaryotic':
        value1 = mat['X'][0][0]
        value2 = mat['X'][2][0]
        data = [value1, value2]
        label = np.squeeze(mat['y'])
    elif dataset == '20NewsGroups':
        mat['data'][0][0], mat['data'][0][1] = mat['data'][0][0].T, mat['data'][0][2].T
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == 'yale_mtv':
        mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = mat['X'][0][0:2]
        # print((data))
        label = np.squeeze(mat['gt'])
    elif dataset == '100leaves':
        mat['data'][0][0], mat['data'][0][1] = mat['data'][0][0].T, mat['data'][0][1].T
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == 'NUSWIDE':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == 'ALOI':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'Wikipedia-test':
        data = mat['X'][0:2][0:2]
        data = np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'Movies':
        data = mat['X'][0:2][0:2]
        data = np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    class_num=len(np.unique(label))
    print(class_num)
    divide_seed = random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), 1-align_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    if dataset == 'Caltech101_7':
        data[0],data[1]=np.squeeze(data[0]),np.squeeze(data[1])
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx],data[0][test_idx], data[1][test_idx]
    '''获取初始训练数据和测试数据'''
    if align_prop != 1:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
    elif align_prop == 1:
        all_data.append(train_X.T)
        all_data.append(train_Y.T)
    traindata.append(train_X.T)
    traindata.append(train_Y.T)
    '''不完整'''
    test_mask = get_sn(2, len(test_label), 1 - complete_prop)
    X_mask, Y_mask = test_mask[:, 0].astype(np.bool_), test_mask[:, 1].astype(np.bool_)
    # test_X[~X_mask] = 0
    # test_Y[~Y_mask] = 0
    test_X, test_Y = test_X[X_mask], test_Y[Y_mask]
    test_label_X, test_label_Y=test_label_X[X_mask], test_label_Y[Y_mask]


    if align_prop != 1:
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))
        all_data.append(np.concatenate((train_X, test_X)).T)
        all_data.append(np.concatenate((train_Y, test_Y)).T)
        all_label = np.concatenate((train_label, test_label))
    elif align_prop == 1:
        all_label_X, all_label_Y = train_label, train_label
        all_label = train_label

    '''构建训练对'''
    view0, view1, noisy_labels, real_labels, _, _ = get_pairs(train_X, train_Y, neg_num, train_label)
    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1
    print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X)), 2))

    if is_noise == 0:  # training with real_labels, v/t with real_labels
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
    else:  # training with labels, v/t with real_labels
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
    '''数据重表示'''
    view0, view1, all_data[0], all_data[1] = torch.from_numpy(view0).float(), torch.from_numpy(
        view1).float(), torch.from_numpy(all_data[0]).float(), torch.from_numpy(all_data[1]).float()
    view0, view1, all_data[0], all_data[1] = np.array(view0), np.array(view1), np.array(all_data[0]), np.array(
        all_data[1])
    train_pairs.append(view0.T)
    train_pairs.append(view1.T)
    train_pair_real_labels = real_labels
    dim = view0.shape[0]
    print(np.shape(view0))
    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, traindata,dim,class_num ,divide_seed

def get_sn(view_num, alldata_len, missing_rate):
    """
    Randomly generate incomplete data information, simulate partial view data with complete view data

    :param view_num: Number of views (模态/视图数量)
    :param alldata_len: Number of samples (样本数量)
    :param missing_rate: Defined in section 4.3 of the paper (缺失率)
    :return: Sn 缺失模式矩阵
    """

    # 将缺失率减半，控制缺失程度
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate  # 数据存在的概率

    # 如果数据存在率很低，确保每个样本至少有一个视图存在
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # 创建独热编码器
        # 生成一个len^view的矩阵，矩阵每一行只有一个1
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve

    error = 1  # 初始化误差

    # 如果没有缺失，直接返回全 1 矩阵
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix

    iteration = 0
    # 循环直到误差低于 0.005
    while error >= 0.005 and iteration < 1000:
        enc = OneHotEncoder()
        # 为每个样本随机保留一个视图
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        # 计算需要保留的 1 的总数（去掉每行已保留的 1）
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)

        # 生成随机缺失矩阵
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # 统计重叠的 1
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))

        # 调整缺失比例
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)

        # 重新生成缺失矩阵
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)

        # print(matrix)

        # 计算误差
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

        iteration += 1

    return matrix
def get_pairs(train_X, train_Y, neg_prop, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])

        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])

    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)

        for k in range(neg_prop):
            view0.append(train_X[j])
            view1.append(train_Y[neg_idx[k]])

            labels.append(0)
            class_labels0.append(train_label[j])
            class_labels1.append(train_label[neg_idx[k]])

            if train_label[j] != train_label[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


class GetpreDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(
            self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)

        return fea0, fea1

    def __len__(self):
        return len(self.data[0].T)
class GetDataset(Dataset):
    def __init__(self, data, labels, real_labels):
        self.data = data
        self.labels = labels
        self.real_labels = real_labels

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(
            self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        real_label = np.int64(self.real_labels[index])
        return fea0, fea1, label, real_label

    def __len__(self):
        return len(self.labels)


class GetAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), \
                           torch.from_numpy(self.data[1][:, index]).float(), \

        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        return fea0, fea1, label, class_labels0, class_labels1

    def __len__(self):
        return len(self.labels)


def loader(train_bs, align_prop, complete_prop,neg_num, is_noise, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param test_prop: known aligned proportions for training MvCLN
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param data_idx: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, traindata,dim, class_num,divide_seed \
        = load_data(align_prop, complete_prop, neg_num, is_noise, dataset)
    train_pair_dataset = GetDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    pretrain_datast=GetpreDataset(traindata)
    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    pretrain_pair_loader=DataLoader(
        pretrain_datast,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False
    )
    return train_pair_loader, all_data, all_label, all_label_X, all_label_Y, pretrain_pair_loader,dim,class_num, divide_seed
