import os, random, sys
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        self.data_views = list()
        self.device = device  # Store device

        # Parse dataset name, extract base name and missing rate
        if '_missing_' in db:
            parts = db.split('_missing_')
            base_db = parts[0]
            missing_ratio = parts[1]
            filename = f'{base_db}_missing_{missing_ratio}.mat'
        else:
            # Default to using 0.1 missing rate
            base_db = db
            missing_ratio = '0.1'
            filename = f'{base_db}_missing_0.1.mat'

        # Record for logging and debugging
        print(f"Loading dataset: {filename}")

        if base_db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, filename))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "MNIST-USPS":
            filename = filename.replace('-', '_')
            mat = sio.loadmat(os.path.join(path, filename))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "BDGP":
            mat = sio.loadmat(os.path.join(path, filename))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "Fashion":
            mat = sio.loadmat(os.path.join(path, filename))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "hand":
            # Fix handwritten dataset filename
            filename = filename.replace('hand_missing', 'handwritten_missing')
            mat = sio.loadmat(os.path.join(path, filename))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y']) + 1).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "scene":
            # Fix Scene15 dataset filename
            filename = filename.replace('scene_missing', 'Scene15_missing')
            mat = sio.loadmat(os.path.join(path, filename))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif base_db == "mfeat":
            # Fix mfeat dataset filename, note that there's no .mat extension here
            filename = filename.replace('.mat', '')
            mat = sio.loadmat(os.path.join(path, filename))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                view_data = X_data[0, idx]
                view_data = np.array(view_data, dtype=np.float32)
                self.data_views.append(view_data)
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        else:
            raise NotImplementedError(f"Dataset {db} not implemented")

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device).float()

        self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        missing_info = self.missing_matrix[index, :]

        return sub_data_views, self.labels[index], missing_info


def get_multiview_data(mv_data, batch_size):
    """
    Create a data loader for multi-view data
    
    Args:
        mv_data: MultiviewData instance
        batch_size: Batch size for the data loader
        
    Returns:
        mv_data_loader: PyTorch DataLoader for the multi-view data
        num_views: Number of views in the dataset
        num_samples: Total number of samples
        num_clusters: Number of unique clusters/classes
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
