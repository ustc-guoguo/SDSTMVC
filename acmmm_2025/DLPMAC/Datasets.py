import random
import numpy as np

import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_list):
        self.X_list = X_list
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        for view in range(self.view_size):
            current_x = self.X_list[view][index]
            current_x_list.append(current_x)

        # permutation
        P_index = random.sample(range(len(index)), len(index))
        P = np.eye(len(index)).astype('float32')
        P = P[:, P_index]
        current_x_list[1] = current_x_list[1][P_index]
        return current_x_list, P

    def __len__(self):
        # return the total size of data
        return self.X_list[0].shape[0]


class Data_Sampler(object):
    """Custom Sampler is required. This sampler prepares batch by passing list of
    data indices instead of running over individual index as in pytorch sampler"""

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size