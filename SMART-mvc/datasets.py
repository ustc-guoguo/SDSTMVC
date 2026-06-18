
import os, sys
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from utils import min_max_normalize, set_random_seed


class myDataset():
    def __init__(self, data_name, root='/mlx_devbox/users/wangminglei.04/playground/mvc/datasets/', aligned_ratio=None, view_list = None):
        self.data_name = data_name
        self.root = root
        self.data_path = os.path.join(self.root, f"{self.data_name}.mat")

        set_random_seed(20)

        self.view_list = view_list
        self.data_list, self.true_label = self.read_data()
        if self.view_list is not None:
            self.n_views = len(view_list)
        else:
            self.n_views = len(self.data_list)
        self.n_samples = len(self.true_label)
        self.n_classes = len(np.unique(self.true_label))
        self.dim_list = []
        for view in range(self.n_views):
            self.dim_list.append(self.data_list[view].shape[-1])

        if aligned_ratio is not None:
            self.aligned_ratio = aligned_ratio
            self.flag, self.indices_v2_all, self.indices_aligned, self.indices_v2_misaligned = self.get_misaligned(self.aligned_ratio)


    def get_misaligned(self, aligned_ratio):
        """
        Randomly allocate aligned and misaligned samples based on the alignment ratio.
            inputs:
                aligned_ratio: float. the alignment ratio
            returns:
                flag: bool. A list with the length of n_samples. True: the views are aligned, False: unaligned.
                indices_v2_all: int. The indices of all samples in view2.
                indices_aligned: int. The indices of only the aligned part samples in both view1 and view2.
                indices_v2_misaligned: int. The indices of only the unaligned part samples in view2.
        """
        n_aligned = int(self.n_samples * aligned_ratio)
        indices = np.random.permutation(self.n_samples)
        flag = indices < 0
        flag[indices[:n_aligned]] = True

        indices_v2_all = np.linspace(0, self.n_samples - 1, self.n_samples, dtype=int)
        # Indices of aligned samples, the same for all views.
        indices_aligned = indices_v2_all[flag]
        # Indices of shuffled and misaligned samples, only for the shuffled view.
        indices_v2_misaligned = shuffle(indices_v2_all[~flag], random_state=2)
        indices_v2_all[~flag] = indices_v2_misaligned
        # P_gt = np.eye(self.n_samples).astype('float32')
        # P_gt = P_gt[:, indices_v2_all]

        return flag, indices_v2_all, indices_aligned, indices_v2_misaligned

    def get_normal_data(self):
        return self.data_list, self.true_label

    def get_aligned_part_data(self):
        '''Only the aligned part samples.'''
        x_view1 = self.data_list[0][self.indices_aligned]
        x_view2 = self.data_list[1][self.indices_aligned]
        label = self.true_label[self.indices_aligned]

        return x_view1, x_view2, label

    def get_misaligned_part_data(self):
        '''Only the unaligned part samples.'''
        x_view1 = self.data_list[0][~self.flag]                     # the unaligned samples in view1
        x_view2 = self.data_list[1][self.indices_v2_misaligned]     # the unaligned samples in view2
        label_view1 = self.true_label[~self.flag]
        label_view2 = self.true_label[self.indices_v2_misaligned]

        return x_view1, x_view2, label_view1, label_view2

    def get_partially_aligned_data(self):
        '''All samples, containing the aligned part and the unaligned part.'''
        x_view1 = self.data_list[0]
        x_view2 = self.data_list[1][self.indices_v2_all]
        label_view2 = self.true_label[self.indices_v2_all]

        return x_view1, x_view2, self.true_label, label_view2

    def load_pretrain_data(self, batch_size=1000):
        data = []
        x_view1, x_view2, label = self.get_aligned_part_data()
        data.append(x_view1)
        data.append(x_view2)
        pretrain_loader = DataLoader(MVDataset(data, label), batch_size=batch_size, shuffle=True)

        return pretrain_loader

    def load_train_data(self, batch_size=1000):
        data, label_list = [], []
        x_view1, x_view2, label_view1, label_view2 = self.get_partially_aligned_data()
        data.append(x_view1)
        data.append(x_view2)
        label_list.append(label_view1)
        label_list.append(label_view2)
        flag = self.flag
        train_loader = DataLoader(MVDatasetAll(data, label_list, flag), batch_size=batch_size, shuffle=True)

        return train_loader

    def read_data(self):
        X_list = []
        label = None
        mat = sio.loadmat(self.data_path)
        if self.data_name in ['Scene-15']:
            X = mat['X'][0]
            X_list.append(X[0].astype('float32'))
            X_list.append(X[1].astype('float32'))
            label = np.squeeze(mat['Y'])

        elif self.data_name in ['HandWritten']:
            X = mat['X'][0]
            views = self.view_list if self.view_list is not None else [0, 2]
            for view in views:
                X_list.append(min_max_normalize(X[view]).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['Caltech101-7', 'Caltech101-20']:
            X = mat['X'][0]
            views = self.view_list if self.view_list is not None else [3, 4]
            for view in views:
                X_list.append(min_max_normalize(X[view]).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['YoutubeFace']:
            # X = mat['X'][0]
            y = np.squeeze(mat['Y']).astype('int')
            idx = np.argsort(y)
            views = self.view_list if self.view_list is not None else [0, 3]
            for view in views:
                X_list.append(min_max_normalize(mat['X'][view, 0][idx]).astype('float32'))
            label = y[idx]

        elif self.data_name in ['BDGP']:
            X_list.append(min_max_normalize(mat['X1']).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['X2']).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['Reuters_dim10']:
            x_train = mat['x_train']
            x_test = mat['x_test']
            y = np.squeeze(np.hstack((mat['y_train'], mat['y_test']))).astype('int')
            idx = np.argsort(y)
            views = self.view_list if self.view_list is not None else [0, 1]
            for view in views:
                x = min_max_normalize(np.vstack((x_train[view], x_test[view])).astype('float32'))
                X_list.append(x[idx])
            label = y[idx]

        elif self.data_name in ['MNIST-USPS']:
            X_list.append(min_max_normalize(mat['X1']).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['X2']).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['NoisyMNIST']:
            X_list.append(min_max_normalize(mat['X1']).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['X2']).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['Deep Animal']:
            X_list.append(min_max_normalize(mat['X'][0, 6].T).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['X'][0, 5].T).astype('float32'))
            label = np.squeeze(mat['gt']).astype('int')

        elif self.data_name in ['CUB']:
            X_list.append(min_max_normalize(mat['X'][0][0]).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['X'][0][1]).astype('float32'))
            label = np.squeeze(mat['gt']).astype('int')

        elif self.data_name in ['Wiki']:
            X_list.append(min_max_normalize(mat['Txt']).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['Img']).astype('float32'))
            label = np.squeeze(mat['label']).astype('int')

        elif self.data_name in ['NUS-WIDE']:
            X_list.append(min_max_normalize(mat['Img']).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(mat['Txt']).astype('float32'))
            label = np.squeeze(mat['label']).astype('int')

        elif self.data_name in ['Hdigit']:
            X = mat['data'][0]
            X_list.append(min_max_normalize(X[1].T.astype('float32')))
            X_list.append(min_max_normalize(X[0].T.astype('float32')))
            label = np.squeeze(mat['truelabel'][0, 0]).astype('int')

        elif self.data_name in ['Fashion']:
            x1 = mat['X1'].reshape(10000, -1)
            x2 = mat['X2'].reshape(10000, -1)
            X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
            X_list.append(min_max_normalize(x2).astype('float32'))
            label = np.squeeze(mat['Y']).astype('int')

        elif self.data_name in ['Cifar10', 'Cifar100']:
            views = self.view_list if self.view_list is not None else [0, 1]
            for view in views:
                X_list.append(min_max_normalize(mat['data'][view, 0].T).astype('float32'))
            label = np.squeeze(mat['truelabel'][0, 0]).astype('int')

        return X_list, label


class MVDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x0 = torch.from_numpy(self.data[0][idx]).float()
        x1 = torch.from_numpy(self.data[1][idx]).float()
        label = self.label[idx]
        return x0, x1, label, torch.from_numpy(np.array(idx)).long()


class MVDatasetAll(Dataset):
    def __init__(self, data, label_list, flag_list):
        self.data = data
        self.label_list = label_list
        self.flag_list = flag_list

    def __len__(self):
        return len(self.label_list[0])

    def __getitem__(self, idx):
        x0 = torch.from_numpy(self.data[0][idx]).float()
        x1 = torch.from_numpy(self.data[1][idx]).float()
        label1 = self.label_list[0][idx]
        label2 = self.label_list[1][idx]
        flag = self.flag_list[idx]
        return x0, x1, label1, label2, flag, torch.from_numpy(np.array(idx)).long()


def load_data(config, root):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    # mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
    mat = sio.loadmat(f"{root}/{config['dataset']}.mat")
    if data_name in ['Scene-15']:
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['HandWritten']:
        X = mat['X'][0]
        for view in [0, 2]:
            x = X[view]
            x = min_max_normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Caltech101-7', 'Caltech101-20']:
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = min_max_normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['BDGP']:
        x1 = mat['X2']
        x2 = mat['X1']
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Reuters_dim10']:
        X = mat['x_train']
        y = np.squeeze(mat['y_train']).astype('int')
        idx = np.argsort(y)
        y = y[idx]
        for view in [0, 1]:
            x = X[view][idx]
            x = min_max_normalize(x).astype('float32')
            X_list.append(x)
        Y_list.append(y)

    elif data_name in ['MNIST-USPS']:
        x1 = mat['X1']
        x2 = mat['X2']
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['NoisyMNIST']:
        x1 = mat['X1']
        x2 = mat['X2']
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Deep Animal']:
        x1 = mat['X1']
        x2 = mat['X2']
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['CUB']:
        x1 = mat['X'][0][0]
        x2 = mat['X'][0][1]
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['gt']).astype('int')
        Y_list.append(y)

    elif data_name in ['WIKI']:
        x1 = mat['Img']
        x2 = mat['Txt']
        X_list.append(min_max_normalize(x1).astype('float32'))  # (1449,2048)
        X_list.append(min_max_normalize(x2).astype('float32'))
        y = np.squeeze(mat['label']).astype('int')
        Y_list.append(y)

    return X_list, Y_list,


def get_aligned(num_sample, aligned_ratio, random_state=2):
    """
    根据样本数量和对齐率进行分配
        inputs:
            num_sample: 样本数量
            aligned_ratio: 对齐率
        returns:
            flag: 长度为n的分配列表，其中 True: 表示该索引位置两个视图的数据是对齐的，False: 表示该索引位置两个视图的数据不是对齐的
    """
    num_aligned = int(num_sample * aligned_ratio)
    # 打乱索引
    index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index = shuffle(index, random_state=random_state)
    # 分配, 取打乱后的前一部分索引位置作为对齐数据
    flag = index < 0
    flag[index[:num_aligned, ]] = True
    return flag


def shuffle_data(x1_train, x2_train, flag, device, random_state=2):
    """
        inputs:
            x1_train: 视图1的数据
            x2_train: 视图2的数据
            flag: 对齐和非对齐的分配，长度为n的分配列表，其中 True: 表示该索引位置两个视图的数据是对齐的，False: 表示该索引位置两个视图的数据不是对齐的
        returns:
            x1: 视图1的数据
            x2: 视图2的数据
            P_index: 打乱的索引
            index_mis_aligned: 非对齐部分的索引
            P_gt: 真实的转换矩阵
    """
    num_sample = x1_train.shape[0]
    P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_mis_aligned = shuffle(P_index[~flag], random_state=random_state)
    P_index[~flag] = index_mis_aligned
    P_gt = np.eye(num_sample).astype('float32')
    P_gt = P_gt[:, P_index]
    P_gt = torch.from_numpy(P_gt).to(device)
    x1 = x1_train
    x2 = x2_train[P_index]
    return x1, x2, P_index, index_mis_aligned, P_gt

def get_mis_aligned(x1_train, x2_train, true_label, flag, device):
    num_sample = x1_train.shape[0]
    # 对齐
    P_index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_mis_aligned = shuffle(P_index[~flag], random_state=2)
    P_index[~flag] = index_mis_aligned
    P_gt = np.eye(num_sample).astype('float32')
    P_gt = P_gt[:, P_index]
    P_gt = torch.from_numpy(P_gt).to(device)
    x1_eval = x1_train
    x2_eval = x2_train[P_index]
    label_list = (true_label, true_label[P_index])
    # x1_label = true_label
    # x2_label = true_label[P_index]
    # x1_eval = x1_train[P_index]
    # x2_eval = x2_train
    return x1_eval, x2_eval, label_list, P_index, index_mis_aligned, P_gt

