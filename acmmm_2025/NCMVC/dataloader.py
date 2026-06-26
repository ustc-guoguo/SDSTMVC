from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import torch.nn.functional as F

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        if path.split('/')[-1] == 'Hdigit.mat':
            self.view1 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))
            self.view2 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
            self.labels = scipy.io.loadmat(path)['truelabel'][0, 0].transpose()
        elif path.split('/')[-1] == 'MNIST-USPS.mat':
            self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
            self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
            self.labels = scipy.io.loadmat(path)['Y'].transpose()
        elif path.split('/')[-1] == 'BDGP.mat':
            self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
            self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
            self.labels = scipy.io.loadmat(path)['Y'].transpose()
        elif path.split('/')[-1] == 'Reuters_dim10.mat':
            self.view1 = scaler.fit_transform(np.vstack((data['x_train'][0], data['x_test'][0])).astype(np.float32))
            self.view2 = scaler.fit_transform(np.vstack((data['x_train'][1], data['x_test'][1])).astype(np.float32))
            self.labels = np.hstack((data['y_train'], data['y_test'])).transpose()
        elif path.split('/')[-1] == 'NUS-WIDE.mat':
            self.view1 = scaler.fit_transform(data['Img'].astype(np.float32))
            self.view2 = scaler.fit_transform(data['Txt'].astype(np.float32))
            self.labels = scipy.io.loadmat(path)['label'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "HandWritten": # handwritten
        dataset = Caltech('data/HandWritten.mat', view=2)
        dims = [240, 216]
        view = 2
        data_size = 2000
        class_num = 10
    elif dataset == "Hdigit": # 
        dataset = Caltech('data/Hdigit.mat', view=2)
        dims = [256, 784]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "MNIST-USPS": # 
        dataset = Caltech('data/MNIST-USPS.mat', view=2)
        dims = [784, 256]
        view = 2
        data_size = 5000
        class_num = 10
    elif dataset == "BDGP": # 
        dataset = Caltech('data/BDGP.mat', view=2)
        dims = [79, 1750]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Reuters_dim10": # 
        dataset = Caltech('data/Reuters_dim10.mat', view=2)
        dims = [10, 10]
        view = 2
        data_size = 18758
        class_num = 6
    elif dataset == "NUS-WIDE": # 
        dataset = Caltech('data/NUS-WIDE.mat', view=2)
        dims = [4096, 300]
        view = 2
        data_size = 9000
        class_num = 10
    
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
