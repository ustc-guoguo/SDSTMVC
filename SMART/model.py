import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cosine_sim



class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space.

    """
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 emb_dim,
                 n_layers_e,
                 n_layers_d,
                 activation='relu',
                 batchnorm=True,
                 ):
        super(Autoencoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation
        self.batchnorm = batchnorm

        encoder_layers = []
        if self.n_layers_e > 1:
            # More than 1 layer
            for i in range(self.n_layers_e):
                if i == 0:
                    encoder_layers.append(nn.Linear(self.in_dim, self.hidden_dim))
                elif i == self.n_layers_e - 1:
                    encoder_layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
                else:
                    encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if i < self.n_layers_e-1:
                    if self.batchnorm:
                        encoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
                    if self.activation == 'sigmoid':
                        encoder_layers.append(nn.Sigmoid())
                    elif self.activation == 'relu':
                        encoder_layers.append(nn.ReLU())
                    elif self.activation == 'leakyrelu':
                        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self.activation == 'tanh':
                        encoder_layers.append(nn.Tanh())
                    elif self.activation == 'none':
                        encoder_layers.append(nn.Identity())
                    else:
                        raise ValueError(f"Unknown activation type {self.activation}")
        else:
            # Only 1 layer
            encoder_layers.append(nn.Linear(self.in_dim, self.emb_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        if self.n_layers_d > 1:
            # More than 1 layer
            for i in range(self.n_layers_d):
                if i == 0:
                    decoder_layers.append(nn.Linear(self.emb_dim, self.hidden_dim))
                elif i == self.n_layers_d-1:
                    decoder_layers.append(nn.Linear(self.hidden_dim, self.in_dim))
                else:
                    decoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if i < self.n_layers_d-1:
                    if self.batchnorm:
                        decoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
                    if self.activation == 'sigmoid':
                        decoder_layers.append(nn.Sigmoid())
                    elif self.activation == 'relu':
                        decoder_layers.append(nn.ReLU())
                    elif self.activation == 'leakyrelu':
                        decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self.activation == 'tanh':
                        decoder_layers.append(nn.Tanh())
                    elif self.activation == 'none':
                        decoder_layers.append(nn.Identity())
                    else:
                        raise ValueError(f"Unknown activation type {self._activation}")
        else:
            # Only 1 layer
            decoder_layers.append(nn.Linear(self.emb_dim, self.in_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        embedding = self.encoder(x)
        return embedding

    def decode(self, embedding):
        x_hat = self.decoder(embedding)
        return x_hat

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = F.normalize(embedding, dim=1, p=2)
        x_hat = self.decoder(embedding)
        return embedding, x_hat


class Model(nn.Module):
    """View specific autoencoders.

    Parameters:
        - in_dim: a list of int. Dimensions of the input features view^1 and views^2.
        - emb_dim: int. Dimension of the latent embeddings.
        - hidden_dim: int. Dimension of the hidden layers in encoder/decoder.
        - n_layers_e: positive int. Number of the encoder layers.
        - n_layers_d: positive int. Number of the decoder layers.
        - activation: activation method. Including "sigmoid", "tanh", "relu", "leakyrelu".
            default: "relu".
        - batchnorm: boolean. It provided whether to use the
            batchnorm in autoencoders.
        - device: the device on which to train the model.
        """
    def __init__(self,
                 in_dim,
                 emb_dim,
                 hidden_dim=2000,
                 n_layers_e=4,
                 n_layers_d=4,
                 activation='leakyrelu',
                 batchnorm=True,
                 device = torch.device('cuda')):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation
        self.batchnorm = batchnorm
        self.device = device

        self.autoencoder1 = Autoencoder(in_dim=self.in_dim[0],
                                        hidden_dim=self.hidden_dim,
                                        emb_dim=self.emb_dim,
                                        n_layers_e=self.n_layers_e,
                                        n_layers_d=self.n_layers_d,
                                        activation=self.activation,
                                        batchnorm=self.batchnorm)
        self.autoencoder2 = Autoencoder(in_dim=self.in_dim[1],
                                        hidden_dim=self.hidden_dim,
                                        emb_dim=self.emb_dim,
                                        n_layers_e=self.n_layers_e,
                                        n_layers_d=self.n_layers_d,
                                        activation=self.activation,
                                        batchnorm=self.batchnorm)

        self.high_level_projector1 = nn.Linear(self.emb_dim, self.emb_dim)

    # 前向传播（x->emb->x_hat）
    def forward(self, X1, X2, is_eval=False):

        emb1 = self.autoencoder1.encode(X1)
        emb2 = self.autoencoder2.encode(X2)

        if is_eval:
            return emb1, emb2
        else:
            X1_hat = self.autoencoder1.decode(emb1)
            X2_hat = self.autoencoder2.decode(emb2)
            return emb1, emb2, X1_hat, X2_hat

    # 高级投影（emb->H）
    def high_level_project(self, emb1, emb2):
        H1 = self.high_level_projector1(emb1)
        H2 = self.high_level_projector1(emb2)

        H1 = F.normalize(H1, dim=1, p=2)
        H2 = F.normalize(H2, dim=1, p=2)

        return H1, H2


    # 1. 重构损失
    def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
        return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)

    # 2. 语义匹配对比损失
    def SMC_loss(self, H1, H2, W, tau=1.0):
        '''Semantic Matching Contrastive Learning'''
        eps = 1e-8

        def _row_normalize(mat):
            mat = torch.clamp(mat, min=0)
            return mat / (torch.sum(mat, dim=1, keepdim=True) + eps)
        logits = cosine_sim(H1, H2, device=self.device) / max(tau, eps)
        log_p = F.log_softmax(logits, dim=1)
        q = _row_normalize(W)
        loss12 = -(q * log_p).sum(dim=1).mean()

        logits_t = logits.t()
        log_p_t = F.log_softmax(logits_t, dim=1)
        q_t = _row_normalize(W.t())
        loss21 = -(q_t * log_p_t).sum(dim=1).mean()

        return 0.5 * (loss12 + loss21)

    # 语义权重矩阵 -> 2
    def get_semantic_graph(self, score, flag, aligned_score, topk=None, mutual=False):
        """
        score: NxN; 每个样本的两个视图特征的相似度矩
        flag: N; 每个样本是否对齐的标志（1表示对齐，0表示未对齐）
        aligned_score: N_a, the number of aligned samples. 对齐样本的跨视图相似度分数
        """
        diag_score = torch.diag(score)
        weights_diag = torch.zeros(diag_score.size(), device=self.device)

        # # 均值 & 标准差 -> 阈值(确定什么样的相似度才算是"语义相关")
        mean_aligned_score = torch.mean(aligned_score)
        std_aligned_score = torch.std(aligned_score)
        threshold = mean_aligned_score - std_aligned_score
        threshold = max(threshold, 0.01)
        
        ### 1. weights_diag 计算对角线元素的权重
        indices_neutral = diag_score >= threshold
        weights_diag[indices_neutral] = diag_score[indices_neutral]
        weights_diag[flag] = 1

        ### 2. 语义图 weighted_adj
        weighted_adj = score - torch.diag_embed(torch.diag(score))
        weighted_adj = torch.where(weighted_adj >= threshold, weighted_adj, 0)
        # =================================================  新增
        if mutual:
            weighted_adj = weighted_adj * (weighted_adj.t() > 0).to(weighted_adj.dtype)

        if topk is not None:
            N = weighted_adj.size(0)
            k = int(topk)
            if k > 0:
                k = min(k, max(N - 1, 1))
                vals, idx = torch.topk(weighted_adj, k=k, dim=1)
                mask = torch.zeros_like(weighted_adj)
                mask.scatter_(1, idx, (vals > 0).to(weighted_adj.dtype))
                weighted_adj = weighted_adj * mask
        # =======================================================
        weighted_adj = weighted_adj + torch.diag_embed(weights_diag)
        1
        ### 3. 确保语义图中不存在全为0的行
        sum_row_ = torch.sum(weighted_adj, dim=-1)
        indices_zero_ = torch.where(sum_row_ == 0)
        if len(indices_zero_[0]) > 0:
            idices_max_ = torch.argmax(score[indices_zero_[0]], dim=-1)
            weighted_adj[indices_zero_[0], idices_max_] = 0.01
        return weighted_adj

    # 3. 视图分布对齐损失
    def VDA_loss(self, emb1, emb2, corr_coef_matrix=None):
        # return F.mse_loss(emb1, emb2)
        '''View Distribution Alignment'''
        # Covariance Matching Alignment
        if corr_coef_matrix is None:
            corr_coef_matrix = self.cross_corr_coef_matrix(emb1, emb2)
        diag_ = torch.diag(corr_coef_matrix)
        ones_ = torch.ones_like(diag_, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(diag_, ones_) # 视图内 --- 每个对象在两个视图中尽可能语义相似

        # TODO: Cross-view Feature Alignment
        corr_coef_matrix_1 = self.cross_corr_coef_matrix(emb1, emb1)
        corr_coef_matrix_2 = self.cross_corr_coef_matrix(emb2, emb2)
        # loss += F.mse_loss(corr_coef_matrix_1, corr_coef_matrix_2) # 视图间 --- 两个视图的特征对齐
        
        # TODO: 视图分布对齐 mse -> KL divergence ( MSE 强迫数值绝对相等，过于严苛且忽略了分布结构 |  KL 散度关注样本间的相对关系分布 (Relational Distribution)，更符合语义对齐的目标)
        tau = 0.5
        log_p1 = F.log_softmax(corr_coef_matrix_1 / tau, dim=1)
        p2 = F.softmax(corr_coef_matrix_2 / tau, dim=1)
        log_p2 = F.log_softmax(corr_coef_matrix_2 / tau, dim=1)
        p1 = F.softmax(corr_coef_matrix_1 / tau, dim=1)
        loss_kl = 0.5 * (F.kl_div(log_p1, p2, reduction='batchmean') + F.kl_div(log_p2, p1, reduction='batchmean'))
        loss += loss_kl
        return loss

    # 相似度矩阵 -> 3
    def cross_corr_coef_matrix(self, emb1, emb2):
        emb1_std = self.feature_standardize(emb1) # torch.Size([2500, 30])
        emb2_std = self.feature_standardize(emb2) # torch.Size([2500, 30]) 
        N, D = emb1.size()
        eps = 1e-5
        eye_ = torch.eye(N, dtype=torch.float32, device=self.device) # 单位矩阵
        cov_matrix = torch.mm(emb1_std, emb2_std.t()) / (D - 1)  + (eps * eye_) # 协方差矩阵(矩阵乘法计算两个标准化嵌入的点积)

        return cov_matrix

    # 特征标准化
    def feature_standardize(self, feature):
        feat_mean = torch.mean(feature.detach(), dim=-1).unsqueeze(-1)
        feat_std = torch.std(feature.detach(), dim=-1).unsqueeze(-1)
        ones_row_vec = torch.ones(feature.size(-1), dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_standardized = (feature - (torch.mm(feat_mean, ones_row_vec))).div(torch.mm(feat_std, ones_row_vec))

        return feat_standardized
