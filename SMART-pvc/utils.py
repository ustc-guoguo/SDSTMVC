
import os
import logging
import random
import numpy as np
import math
import torch
import torch.nn.functional as F
import scipy.io as sio
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def get_fusion(Z1, Z2, ratio = 0.5, fusion_mode = 0):
    """
    fusion_mode: \n
    0: (\\alpha * Z1 + (1-\\alpha) * Z2);
    1: Z1 + Z2
    2: concat(Z1, Z2);
    """

    if fusion_mode == 0:
        fusion = ratio * Z1 + (1 - ratio) * Z2
        # fusion = z1 + z2
    elif fusion_mode == 1:
        fusion = Z1 + Z2

    elif fusion_mode == 2:
        fusion = torch.concat((Z1, Z2), dim=-1)

    return fusion


def next_batch_aligned(x1_train, x2_train, flag, batch_size, device='cuda'):
    num_sample = x1_train.shape[0]
    index = np.linspace(0, num_sample - 1, num_sample, dtype=int)
    index_aligned, index_mis_aligned = shuffle(index[flag]), shuffle(index[~flag])
    num_aligned, num_mis_aligned = len(index_aligned), len(index_mis_aligned)
    # 对齐和未对齐的数据
    x1_aligned, x2_aligned = x1_train[index_aligned], x2_train[index_aligned]
    x1_mis_aligned, x2_mis_aligned = x1_train[index_mis_aligned], x2_train[index_mis_aligned]
    P_index = shuffle(np.linspace(0, num_mis_aligned - 1, num_mis_aligned, dtype=int))
    x2_mis_aligned = x2_mis_aligned[P_index]
    # 循环次数
    total = math.ceil(num_aligned / batch_size)
    # 未对齐的每次取的个数
    batch_size_mis_aligned = math.ceil(num_mis_aligned / total)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(num_aligned, end_idx)
        start_idx1 = i * batch_size_mis_aligned
        end_idx1 = (i + 1) * batch_size_mis_aligned
        end_idx1 = min(num_mis_aligned, end_idx1)
        # 批量对齐数据
        batch_x1_aligned = x1_aligned[start_idx: end_idx, ...]
        batch_x2_aligned = x2_aligned[start_idx: end_idx, ...]
        batch_x1_mis_aligned = x1_mis_aligned[start_idx1: end_idx1, ...]
        batch_x2_mis_aligned = x2_mis_aligned[start_idx1: end_idx1, ...]
        yield (batch_x1_aligned, batch_x2_aligned, batch_x1_mis_aligned, batch_x2_mis_aligned, (i + 1))


def cal_std(logger, accumulated_metrics):
    """Return the average and its std"""
    logger.info('ACC:'+ str(accumulated_metrics['acc']))
    logger.info('NMI:'+ str(accumulated_metrics['nmi']))
    logger.info('ARI:'+ str(accumulated_metrics['ari']))
    output = """ ACC {:.4f} NMI {:.4f} ARI {:.4f}""".format(
        np.mean(accumulated_metrics['acc']),
        np.mean(accumulated_metrics['nmi']),
        np.mean(accumulated_metrics['ari']))
    logger.info(output)

    return


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


def euclidean_distance_np(data1, data2, lack_memory=False):
    """data1, data2: numpy.ndarray"""
    m, d1 = data1.shape
    n, d2 = data2.shape
    assert d1 == d2, f"data1 and data2 must have the same column dimension, but got different: data1{data1.shape}, data2{data2.shape}!"

    if lack_memory:
        dis = []
        for i in tqdm(range(m)):
            # 1xd1
            dis_one2n = data1[i] - data2
            dis_one2n = np.sum(dis_one2n**2, axis=-1)
            dis.append(dis_one2n)
        dis = np.squeeze(np.array(dis))
    else:
        # N*1*M
        A = data1.reshape(m, 1, d1)
        # 1*N*M
        B = data2.reshape(1, n, d2)

        dis = (A - B) ** 2.0
        # N*N matrix for pairwise euclidean distance
        dis = np.squeeze(dis.sum(axis=-1))

    return dis


def cosine_sim(data1, data2, device=torch.device('cuda:0')):
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


def cosine_distance(data1, data2, device=torch.device('cuda:0')):
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


def kl_div_matrix(matrix1, matrix2, tao_kl=1.0, device=torch.device('cuda:0')):
    if matrix1.device != device or matrix2.device != device:
        matrix1, matrix2 = matrix1.to(device), matrix2.to(device)

    matrix1 = F.softmax(matrix1 / tao_kl, dim=-1)
    matrix2 = F.softmax(matrix2 / tao_kl, dim=-1)

    n_samples = matrix1.size(0)
    # part1 = torch.sum(matrix1 * torch.log(matrix1), dim=-1).view(n_samples, 1)
    # part2 = torch.matmul(matrix1, torch.log(matrix2).t())

    kl_div_matrix = torch.sum(matrix1 * torch.log(matrix1), dim=-1).view(n_samples, 1)
    kl_div_matrix = kl_div_matrix.expand(n_samples, n_samples) - torch.matmul(matrix1, torch.log(matrix2).t())

    # Initialize matrix for storing KL divergences
    # kl_div_matrix = torch.zeros(matrix1.size(0), matrix2.size(0), device=device)

    # Memory limited. Too slow.
    # for i, row1 in enumerate(matrix1):
    #     for j, row2 in enumerate(matrix2):
    #         kl_div_matrix[i, j] = (row1 * (torch.log(row1) - torch.log(row2))).sum()

    # Memory limited. Slow.
    # for i, row1 in enumerate(matrix1):
    #     kl_div_matrix[i, :] = (row1 * (torch.log(row1) - torch.log(matrix2))).sum(dim=-1)
    #     # kl_div_matrix[i, :] = torch.matmul(row1, (torch.log(row1) - torch.log(matrix2)).t())

    # Compute KL divergence using matrix multiplication
    # kl_div_matrix = torch.einsum('nid,njd->ij', matrix1[:, None],
    #                                  torch.log(matrix1[:, None]) - torch.log(matrix2[None, :]))
    # kl_div_matrix = torch.matmul(matrix1.unsqueeze(1), (matrix1.unsqueeze(1) - matrix2.unsqueeze(0)).permute(0, 2, 1)).squeeze()

    # Memory sufficient
    # kl_div_inter_samples = F.kl_div(torch.unsqueeze(dis_z_stu_log, dim=1), torch.unsqueeze(dis_z_tch, dim=0), reduction='none')
    # kl_div_inter_samples = torch.sum(kl_div_inter_samples, dim=-1)

    return kl_div_matrix


def pseudo_graph(label_pred, device):
    label_pred = torch.tensor(label_pred).to(device)
    pseudo_g = (label_pred == label_pred.unsqueeze(1)).float()
    # pseudo_g = pseudo_g - torch.eye(pseudo_g.size()[0], device=device)

    diag = torch.diag(pseudo_g)
    pseudo_g = pseudo_g - torch.diag_embed(diag)

    return pseudo_g


def high_confidence_adj(emb1, emb2, label_pred, dis, adj, k=0.1, device=torch.device('cuda:0')):
    # the distance of each node to the center of the class it is assigned to.
    dis = torch.min(dis, dim=-1).values
    # indices of non-high-confidence label
    values, indices = torch.topk(dis, int(len(dis) * (1 - k)), largest=True)

    pseudo_adj = pseudo_graph(label_pred, device)
    # delete connections of nodes with non-high-confidence label
    pseudo_adj[:, indices] = 0
    pseudo_adj[indices, :] = 0

    sim_matrix = cosine_sim(emb1, emb2, device=device)
    # diag = torch.diag(sim_matrix)
    # sim_matrix = sim_matrix - torch.diag_embed(diag)

    s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
    s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
    sim_matrix = (sim_matrix - s_min) / (s_max - s_min)

    sim_matrix = torch.where(adj == 1, sim_matrix, 0)
    pseudo_adj = pseudo_adj + sim_matrix

    # Limit the maximum value to 1
    pseudo_adj = torch.clamp(pseudo_adj, max=1)

    return pseudo_adj


def pseudo_labels(label_pred, centers, dis, k, device=torch.device('cuda:0')):
    label_pred = torch.tensor(label_pred, device=device)
    high_confidence = torch.min(dis, dim=1).values
    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * k)]
    # high_confidence_idx = np.argwhere(high_confidence <= threshold)[0]
    # h_i = high_confidence_idx.numpy()
    high_confidence_idx = torch.where(high_confidence <= threshold)[0]
    pseudo_labels = label_pred[high_confidence_idx]         # (1 x N*k)

    return pseudo_labels, high_confidence_idx


def pseudo_neighbor(emb1, emb2, adj, k=2, device=torch.device('cuda')):
    sim_matrix = cosine_sim(emb1, emb2, device=device)
    diag = torch.diag(sim_matrix)
    sim_matrix = sim_matrix - torch.diag_embed(diag)

    values, indices = torch.topk(sim_matrix, k, dim=-1)  # top k neighbor
    pseudo_adj = torch.zeros_like(sim_matrix, device=device)
    pseudo_adj = pseudo_adj.scatter_(1, indices, 1)  # 在每行中按照列索引将值 scatter 到目标张量中

    # pseudo_adj = pseudo_adj + adj
    # pseudo_adj = torch.where(pseudo_adj == 2, pseudo_adj, 1)

    return pseudo_adj


def add_gaussian_noise(X, device=torch.device("cuda:0")):
    """
    Add gaussian noise to the feature matrix X
    Params
        - X: the feature matrix
    Return
        the noised feature matrix X_tilde
    """
    Noise = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    X_tilde1 = torch.mul(X, Noise)

    return X_tilde1


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

def to_tensor(x, device='cuda'):
    return torch.from_numpy(x.astype(np.float32)).to(device)

def to_numpy(x):
    return x.cpu().numpy()


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


def plot_loss(loss_dict, epochs, save_name='loss.png'):
    plt.plot(loss_dict['loss'], color='#d62728')
    plt.title("loss")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def plot_metrics(score_dict, epochs, save_name='scores.png'):
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    plt.plot(score_dict['acc'], color=colors[0])
    plt.plot(score_dict['nmi'], color=colors[1])
    plt.plot(score_dict['ari'], color=colors[2])
    # plt.plot(score_list['pur'], color=colors[3])
    plt.title("scores")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def show_tsne(feature, true_label, random_state=5, save_name=None):
    mpl.rcParams.update({'figure.dpi': 150})
    colors14 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a','#0687aa']

    n_clusters = len(np.unique(true_label))
    cmap = ListedColormap(colors14[:n_clusters])

    X_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(feature)
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0],
                X_tsne[:, 1],
                c=true_label,
                label="feature",
                s=45,
                marker='o',
                cmap=cmap)
    # plt.legend(fontsize = 20)
    # plt.legend(fontsize = 20)
    # plt.colorbar()  # 显示颜色条

    ax = plt.gca()
    # 去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=False, left=False)
    plt.xticks([])
    plt.yticks([])

    # ax.tick_params(labelsize=24)
    # plt.xticks()
    # plt.yticks()

    # plt.tight_layout()
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()


def t_sne_plot(embedding, label, data_name='cora', sample_num=2000, result_path="", random_state=5, show_fig=True):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return fig: figure
    """

    # sampling
    sample_index = np.random.randint(0, embedding.shape[0], sample_num)
    sample_embedding = embedding[sample_index]
    sample_labels = label[sample_index]

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=random_state)
    ts_embedding = ts.fit_transform(sample_embedding[:, :])

    # remove outlier
    mean, std = np.mean(ts_embedding, axis=0), np.std(ts_embedding, axis=0)
    for i in range(len(ts_embedding)):
        if (ts_embedding[i] - mean < 3 * std).all():
            np.delete(ts_embedding, i)

    # normalization
    x_min, x_max = np.min(ts_embedding, 0), np.max(ts_embedding, 0)
    norm_ts_embedding = (ts_embedding - x_min) / (x_max - x_min)

    # plot
    fig = plt.figure()
    for i in range(norm_ts_embedding.shape[0]):
        plt.text(norm_ts_embedding[i, 0], norm_ts_embedding[i, 1], str(sample_labels[i]),
                 color=plt.cm.Set1(sample_labels[i] % 7),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE', fontsize=14)
    plt.axis('off')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        # print(f'{result_path} is not existed')
        # print(f'Creating {result_path}')
    fig_name = f"{data_name}_t-sne_{random_state}.png"
    plt.savefig(f"{result_path}/{fig_name}", bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()

    return fig


def show_heat_map(matrix, label_list, P_index = None, title=None, save_path='heatmap.jpg'):
    """
    Show heat map of matrix with dimension of N x N.

    Parameters
    - matrix: N x N matrix.
    - label: 1 x N matrix, for sorting 'matrix' on rows and columns.
    - P_index: 1 x N matrix, partially aligned index.
    - title: title of the heat map.
    - save_path: the save path of the heat map.
    """
    plt.rc('font', family='Times New Roman')

    if isinstance(matrix, torch.Tensor):
        if matrix.device == torch.device('cuda:0'):
            matrix = matrix.detach().cpu()
        matrix = matrix.numpy()
    data = matrix

    # Sorting on x axis and y axis.
    data = data[np.argsort(label_list[0]), :]
    data = data[:, np.argsort(label_list[1])]
    # if P_index is None:
    #     data = data[:, np.argsort(label)]
    # else:
    #     data = data[:, np.argsort(label[P_index])]

    plt.figure(figsize=(10, 10))
    sns.heatmap(data=data, cmap='YlOrRd')  # 矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签

    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # plt.show()


def show_heat_map_aligned(matrix, label_true, title=None, save_path='heatmap.png'):
    """
    Show heat map of matrix with dimension of N x N.

    Parameters
    - matrix: N x N matrix.
    - label: 1 x N matrix, for sorting 'matrix' on rows and columns.
    - P_index: 1 x N matrix, partially aligned index.
    - title: title of the heat map.
    - save_path: the save path of the heat map.
    """
    plt.rc('font', family='Times New Roman')

    if isinstance(matrix, torch.Tensor):
        if matrix.device == torch.device('cuda:0'):
            matrix = matrix.detach().cpu()
        matrix = matrix.numpy()
    data = matrix

    # Sorting on x axis and y axis.
    sorted_idx = np.argsort(label_true)
    # print(f"data: {data.shape}")            # data: (1250, 1250)
    # print(f"label_true: {label_true}")      # label_true: [2 4 4 ... 4 4 3]
    # print(f"sorted_idx: {sorted_idx}")      # sorted_idx: [475 186 465 ... 199 793 147]
    data = data[sorted_idx, :]
    data = data[:, sorted_idx]
    # if P_index is None:
    #     data = data[:, np.argsort(label)]
    # else:
    #     data = data[:, np.argsort(label[P_index])]

    plt.figure(figsize=(10, 10))
    sns.heatmap(data=data, cmap='YlOrRd')  # YlOrRd OrRd Blues

    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # plt.show()


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
