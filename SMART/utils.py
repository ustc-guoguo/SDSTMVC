
import os
import logging
import random
import numpy as np
import torch



def euclidean_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # N*N matrix for pairwise euclidean distance
    dis = dis.sum(dim=-1).squeeze()

    return dis


def cosine_sim(data1, data2, device=torch.device('cuda')):
    """
    Calculate the cosine similarity between each row in matrix A and that in matrix B.\n
    计算矩阵 A 中的每行与矩阵 B 中的每行之间的余弦相似度

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_sim (Tensor): (N x N) cosine similarity between each row in matrix data1 and\\
        that in matrix data2.
    """
    # if isinstance(data1, torch.Tensor) and data1.device != torch.device('cuda:0'):
    #     data1 = data1.to(device)
    # if isinstance(data2, torch.Tensor) and data2.device != torch.device('cuda:0'):
    #     data2 = data2.to(device)
    data1 = data1.to(device)
    data2 = data2.to(device)

    # unitization eg.[3, 4] -> [3/sqrt(9 + 16), 4/sqrt(9 + 16)] = [3/5, 4/5]
    data1_norm = data1 / torch.norm(data1, dim=1, keepdim=True)
    data2_norm = data2 / torch.norm(data2, dim=1, keepdim=True)

    cos_sim = torch.mm(data1_norm, data2_norm.t())

    # OR
    # A_norm = A / A.norm(dim = -1, keepdim = True)
    # B_norm = B / B.norm(dim = -1, keepdim = True)
    # cos_sim = (A_norm * B_norm).sum(dim=-1)

    return cos_sim


def cosine_distance(data1, data2, device=torch.device('cuda')):
    """\
    Calculate the cosine distance between each row in matrix A and that in matrix B.

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_distance (Tensor): (N x N) cosine distance between each row in matrix data1 and\\
        that in matrix data2.
    """
    cos_sim = cosine_sim(data1, data2, device)
    cos_distance = 1 - cos_sim

    return cos_distance


def min_max_normalize(x):
    """Min-Max Normalize"""
    if isinstance(x, torch.Tensor):
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x = (x - x_min) / (x_max - x_min)

    elif isinstance(x, np.ndarray):
        # x = (x - np.min(x, axis=-1)) / (np.max(x, axis=-1) - np.min(x, axis=-1))
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


def compare_score(score, best_score):
    (acc, nmi, ari, pur, is_best) = score
    (best_acc, best_nmi, best_ari, best_pur, _) = best_score
    is_best = False
    if acc > best_acc:
        is_best = True
        best_score = (acc, nmi, ari, pur, is_best)
    elif acc == best_acc and (nmi >= best_nmi or ari >= best_ari):
        is_best = True
        best_score = (acc, nmi, ari, pur, is_best)
    else:
        best_score = (best_acc, best_nmi, best_ari, best_pur, is_best)
    return best_score


def get_logger(root = './training_logs', filename = None):
    """
    Get logger.

    Parameters
    - root: str. Root directory of log files.
    - filename: str, Optional. The name of log files.

    return
    - logger: Logger
    """
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '[%(asctime)s]%(levelname)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S')

    if filename is not None:
        """Save logs to file"""
        if not os.path.exists(root):
            os.makedirs(root)

        # mode = 'w', overwriting the previous content; 'a', appended to previous file.
        fh = logging.FileHandler(os.path.join(root, filename), "a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    """Print logs at terminal"""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
