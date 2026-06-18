
import numpy as np
import torch
from sklearn.cluster import KMeans
from utils import cosine_distance, euclidean_distance


def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    A = data1.unsqueeze(dim=1)              # N*1*M
    B = data2.unsqueeze(dim=0)              # 1*N*M

    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()         # N*N matrix for pairwise distance

    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.unsqueeze(dim=1)              # N*1*M
    B = data2.unsqueeze(dim=0)              # 1*N*M

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()       # N*N matrix for pairwise distance

    return cosine_dis


def initialize(X, n_clusters): # 随机选择 n_clusters 个样本点作为初始聚类中心
    """
    Initialize cluster centers.

    Parameters
    - X: (torch.tensor) matrix
    - n_clusters: (int) number of clusters
    
    Return
    - initial state: (np.array) 
    """
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    initial_state = X[indices]

    return initial_state


def k_means(X, n_clusters, distance='euclidean', tol=1e-4, device=torch.device('cpu')):
    """
    Perform k-means algorithm on X.

    Parameters
    - X: torch.tensor. matrix
    - n_clusters: int. number of clusters
    - distance: str. pairwise distance 'euclidean'(default) or 'cosine'
    - tol: float. Threshold 
    - device: torch.device. Running device
    
    Return
    - choice_cluster: torch.tensor. Predicted cluster ids.
    - initial_state: torch.tensor. Predicted cluster centers.
    - dis: minimum pair wise distance.
    """
    if distance == 'euclidean':
        # pairwise_distance_function = pairwise_distance
        pairwise_distance_function = euclidean_distance
    elif distance == 'cosine':
        # pairwise_distance_function = pairwise_cosine
        pairwise_distance_function = cosine_distance
    else:
        raise NotImplementedError(f"Not implemented '{distance}' distance!")

    # convert to float
    X = X.float()
    # transfer to device
    X = X.to(device)

    # initialize
    dis_min = float('inf')
    initial_state_best = initialize(X, n_clusters) # 初始化聚类中心
    
    for i in range(20):
        initial_state = initialize(X, n_clusters)
        dis = pairwise_distance_function(X, initial_state).sum() # 计算每个数据点到每个聚类中心的距离
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state
    initial_state = initial_state_best

    
    # 迭代更新聚类中心，直到聚类中心不再改变或达到最大迭代次数
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1) # 计算每个数据点与所有聚类中心的距离后，选择距离最近的聚类中心，并将数据点分配到该聚类

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break
    
    # choice_cluster：一个长度为 N 的数组，表示每个数据点的聚类标签（每个数据点所属的簇的编号）。
    # initial_state：聚类中心（每个簇的中心点，形状为 K x M，其中 K 是聚类数，M 是每个数据点的维度）。
    # dis：每个数据点与聚类中心之间的距离。
    
    return choice_cluster, initial_state, dis


def clustering(feature, cluster_num, seed = 10, device = torch.device('cpu')):
    """
    Clustering using feature matrix.

    Parameters
    - feature: feature matrix. [num_samples, feature_dim]
    - cluster_num: number of clusters.
    - seed: random seed.
    - device: torch.device. device where the clustering algorithm will be running on.

    Return
    - label_pred: predicted label.
    """
    if device == torch.device('cuda'):
        # Perform K-means Clustering on GPU
        label_pred, centers, dis = k_means(X=feature,
                                    n_clusters=cluster_num, 
                                    distance="cosine",           # cosine euclidean
                                    device=device)
        label_pred = label_pred.cpu().numpy()
        return label_pred, centers, dis
    else:
        # Perform K-means Clustering on CPU
        feature.cpu().numpy()
        kmeans = KMeans(n_clusters=cluster_num, 
                        random_state=seed, 
                        init='k-means++')
        label_pred = kmeans.fit_predict(feature)
        return label_pred
