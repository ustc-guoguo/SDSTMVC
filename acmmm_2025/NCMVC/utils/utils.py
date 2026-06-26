import torch
import faiss
import numpy as np
from collections import defaultdict
from time import perf_counter
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import normalize


import faiss
import torch
import numpy as np
from scipy import sparse

# 先补充必须的依赖函数（如果你的代码里没有）
def fetch_normalization(type):
    def norm_adj(adj):
        adj = sparse.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    if type == 'NormAdj':
        return norm_adj
    else:
        raise ValueError(f"不支持的归一化类型: {type}")

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 最终版build_affinity_matrix（无任何报错）
def build_affinity_matrix(X, k, device='cuda'):
    """
    最终稳定版：修复faiss/GPU/梯度/numpy转换所有问题
    """
    # ========== 1. 核心修复：梯度张量转numpy ==========
    if isinstance(X, torch.Tensor):
        # 先detach()再转numpy，解决requires grad问题
        X = X.detach().float().to(device)  # 关键：detach()移除梯度
    else:
        X = torch.tensor(X, dtype=torch.float32, device=device)
    
    # 清洗异常值 + 转numpy（faiss要求）
    X_np = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).cpu().numpy()
    n_samples, n_dim = X.shape

    # ========== 2. CPU版faiss（彻底避开GPU接口） ==========
    faiss.omp_set_num_threads(8)
    index = faiss.IndexFlatL2(n_dim)
    index.add(X_np)
    
    # 防止k超过样本数
    search_k = min(k + 1, n_samples)
    _, ind = index.search(X_np, search_k)

    # ========== 3. 距离计算（修复梯度/索引问题） ==========
    dist = []
    for i in range(n_samples):
        neighbors = ind[i][1:search_k]  # 跳过自身
        if len(neighbors) == 0:
            dist.append(np.array([]))
            continue
        # 关键：X已经detach，可安全转numpy
        xi = X[i].cpu().numpy()
        x_neighbors = X[neighbors].cpu().numpy()
        dist_i = np.linalg.norm(xi - x_neighbors, axis=1)
        dist.append(dist_i)
    
    # ========== 4. 高斯核 + 邻接矩阵（保留原逻辑） ==========
    # 处理空距离的情况
    dist_flat = np.concatenate([d for d in dist if len(d) > 0]) if dist else np.array([])
    aff = torch.exp(-torch.tensor(dist_flat, device=device) ** 2 / 2)
    
    # 构建对称邻接矩阵
    W = torch.zeros(n_samples, n_samples, device=device)
    aff_idx = 0
    for i in range(n_samples):
        neighbors = ind[i][1:search_k]
        n_neighbors = len(neighbors)
        if n_neighbors > 0:
            W[i, neighbors] = aff[aff_idx:aff_idx+n_neighbors]
            W[neighbors, i] = aff[aff_idx:aff_idx+n_neighbors]
            aff_idx += n_neighbors

    # ========== 5. 归一化（保留原逻辑） ==========
    adj = W.cpu().numpy()
    adj_normalizer = fetch_normalization('NormAdj')
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
    ind = torch.from_numpy(ind).to(device)
    
    return adj

def graph_fusion(graphs,num_iters=20, tol=1e-6,device='cuda'):
    A = sum(graphs) / len(graphs)  
    V = len(graphs)  
    N = graphs[0].shape[0]  

    for iteration in range(num_iters):
        previous_A = A.clone()
        weights = []
        for v in range(V):
            dist = torch.norm(A.to_dense() - graphs[v], p='fro')
            weight = 1.0 / (dist + 1e-8)
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32, device=A.device)  
        weights = weights / weights.sum()
        A = torch.zeros((N, N), dtype=torch.float32, device=A.device) 
        for v in range(V):
            A += weights[v] * graphs[v]
        degree = A.to_dense().sum(dim=1)  
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-8))  
        A = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt  
        if torch.norm(A - previous_A, p='fro') < tol:
            break
    indices = torch.stack(A.nonzero(as_tuple=True), dim=0)  
    values = A[indices[0], indices[1]]  
    A = torch.sparse_coo_tensor(indices, values, size=(N, N), device=A.device)
    return A, weights

def normalized_adjacency(adj):       
   adj = adj                        
   adj = sp.coo_matrix(adj)         
   row_sum = np.array(adj.sum(1))   
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()  
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)           
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def aug_normalized_adjacency(adj):      
   adj = adj + sp.eye(adj.shape[0])   
   adj = sp.coo_matrix(adj)            
   row_sum = np.array(adj.sum(1))     
   d_inv_sqrt = np.power(row_sum, -0.5).flatten() 
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)          
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()  

def fetch_normalization(type):  
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  
       'NormAdj': normalized_adjacency,  
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")  
   return func  

def sparse_mx_to_torch_sparse_tensor(sparse_mx):      
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)      
    return torch.sparse.FloatTensor(indices, values, shape)        