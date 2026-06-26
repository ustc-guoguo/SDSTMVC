import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import scipy.sparse as sp


class NUS_WIDE(Dataset): # 未修改
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'nuswide_deep_2_view.mat')['Img'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'nuswide_deep_2_view.mat')['Txt'].astype(np.float32)
        labels = scipy.io.loadmat(path+'nuswide_deep_2_view.mat')['label'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]



class MINIST_USPS(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'MNIST-USPS.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'MNIST-USPS.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'MNIST-USPS.mat')['Y'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class Reuters_dim10(Dataset):
    def __init__(self, path):
        data1 = np.vstack((scipy.io.loadmat(path+'Reuters_dim10.mat')['x_train'][0].astype(np.float32), scipy.io.loadmat(path+'Reuters_dim10.mat')['x_test'][0].astype(np.float32)))
        data2 = np.vstack((scipy.io.loadmat(path+'Reuters_dim10.mat')['x_train'][1].astype(np.float32), scipy.io.loadmat(path+'Reuters_dim10.mat')['x_test'][1].astype(np.float32)))
        labels = np.hstack((scipy.io.loadmat(path+'Reuters_dim10.mat')['y_train'].transpose(), scipy.io.loadmat(path+'Reuters_dim10.mat')['y_test'].transpose()))
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]




def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot

def load_data(dataset):

    path = './dataset/'

    if dataset == "3Sources":
        dataset = ThreeSources(path+'/3sources.mat')
        dims = [3560, 3631,3068]
        view = 3
        data_size = 169
        class_num = 6
    elif dataset == "BBCSport":
        dataset = BBCSport(path+'/BBCSport.mat')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5

    elif dataset == "BDGP":
        dataset = BDGP(path+'/')
        dims = [79, 1750]
        view = 2
        data_size = 2500
        class_num = 5
    
    elif dataset == "HandWritten":
        dataset = HandWritten(path+'/')
        dims = [240, 216]
        view = 2
        data_size = 2000
        class_num = 10
    
    elif dataset == "WIKI":
        dataset = WIKI(path+'/')
        dims = [10, 128]
        view = 2
        data_size = 2866
        class_num = 10

    elif dataset == "Reuters_dim10":
        dataset = Reuters_dim10(path+'/')
        dims = [10, 10]
        view = 2
        data_size = 18758
        class_num = 6

    elif dataset == "MNIST-USPS":
        dataset = MINIST_USPS(path+'/')
        dims = [784, 256]
        view = 2
        data_size = 5000
        class_num = 10

    elif dataset == "NUS-WIDE":
        dataset = NUS_WIDE(path+'/')
        dims = [4096, 300]
        view = 2
        data_size = 9000
        class_num = 10

    elif dataset == "Deep Animal":
        dataset = Deep_Animal(path+'/')
        dims = [4096, 4096]
        view = 2
        data_size = 10158
        class_num = 50
    
    elif dataset == "Hdigit":
        dataset = Hdigit(path+'/')
        dims = [256, 784]
        view = 2
        data_size = 10000
        class_num = 10

    else:
        raise NotImplementedError

    return dataset, dims, view, data_size, class_num


def get_data(data):
    num_view = data[2]
    assert len(data[0].feature_list) == num_view
    feature_list = []
    for i in range(num_view):
        feature_list.append(torch.from_numpy(data[0].feature_list[i]))
        print(data[0].feature_list[i].shape)
        print(data[-2])
        assert data[0].feature_list[i].shape[0] == data[-2]
    y = data[0].label
    n_clusters = len(np.unique(y))
    assert n_clusters == data[-1]
    feat_dims = data[1]

    return feature_list,num_view,feat_dims, y, n_clusters



class ThreeSources(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)

        self.feature = data['data'][0]
        for i in range(self.feature.shape[0]):
            self.feature[i] = self.feature[i].T
        self.x1 = self.feature[0].astype(np.float32)
        self.x2 = self.feature[1].astype(np.float32)        
        self.x3 = self.feature[2].astype(np.float32)        
        self.label = data['truelabel'][0][0].squeeze()
        self.feature_list = [self.x1,self.x2,self.x3]

        for i in range(len(self.feature_list)): 
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class BBCSport(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        self.data1 = sp.csr_matrix(data['X'][0, 0])
        self.data2 = sp.csr_matrix(data['X'][0, 1])
        self.label = data['Y'].flatten()
        self.feature_list = [self.data1,self.data2]
        for i in range(len(self.feature_list)): 
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]




class HandWritten(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'HandWritten.mat')['X'][0][0].astype(np.float32)
        data2 = scipy.io.loadmat(path+'HandWritten.mat')['X'][0][2].astype(np.float32)
        labels = scipy.io.loadmat(path+'HandWritten.mat')['Y'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class WIKI(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'WIKI.mat')['Txt'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'WIKI.mat')['Img'].astype(np.float32)
        labels = scipy.io.loadmat(path+'WIKI.mat')['label'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class Deep_Animal(Dataset): # 未修改
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'DeepAnimal.mat')['X'][0, 6].T.astype(np.float32)
        data2 = scipy.io.loadmat(path+'DeepAnimal.mat')['X'][0, 5].T.astype(np.float32)
        labels = scipy.io.loadmat(path+'DeepAnimal.mat')['gt'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]

class Hdigit(Dataset): # 未修改
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Hdigit.mat')['data'][0][1].T.astype(np.float32)
        data2 = scipy.io.loadmat(path+'Hdigit.mat')['data'][0][0].T.astype(np.float32)
        labels = scipy.io.loadmat(path+'Hdigit.mat')['truelabel'][0, 0].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]