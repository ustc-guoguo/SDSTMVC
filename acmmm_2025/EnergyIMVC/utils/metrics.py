import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from collections import Counter


def calculate_clustering_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels

    Returns:
        Clustering accuracy score
    """
    y_true = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def calculate_purity(y_true, y_pred):
    """
    Calculate clustering purity

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels

    Returns:
        Clustering purity score
    """
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = np.array(y_pred)

    total = 0
    for cluster in np.unique(y_pred):
        indices = np.where(y_pred == cluster)[0]
        if len(indices) == 0:
            continue
        most_common = Counter(y_true[indices]).most_common(1)[0][1]
        total += most_common

    return total / len(y_pred)