import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluation
from utils import cosine_sim, min_max_normalize, add_gaussian_noise


# class Encoder(nn.Module):
#     def __init__(self, in_feature, out_feature):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(in_feature, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 2000),
#             nn.ReLU(),
#             nn.Linear(2000, out_feature),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)
#
# class Decoder(nn.Module):
#     def __init__(self, in_feature, out_feature):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(in_feature, 2000),
#             nn.ReLU(),
#             nn.Linear(2000, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, out_feature)
#         )
#
#     def forward(self, x):
#         return self.decoder(x)
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, in_feature, out_feature):
#         super(CrossAttentionModule, self).__init__()
#         self.in_feature = in_feature
#         self.out_feature = out_feature
#         self.scale = out_feature ** (-0.5)
#         # self.scale = torch.pow(out_feature, 0.5)
#
#         self.Wq = nn.Linear(self.in_feature, self.out_feature)
#         self.Wk = nn.Linear(self.in_feature, self.out_feature)
#         # self.Wv = nn.Linear(self.in_feature, self.out_feature)
#
#     def forward(self, x1, x2):
#         x_q = self.Wq(x1)
#         x_k = self.Wk(x2)
#         # x_v = self.Wv(x2)
#         att_score = torch.matmul(x_q, torch.transpose(x_k, 0, 1))
#         # att_score = att_score * self.scale
#         att_score = F.softmax(att_score, dim=-1)
#         # out = torch.matmul(att_score, x_v)
#
#         return att_score
#
# class ConsistencyGraphModule(nn.Module):
#     def __init__(self, in_feature, out_feature, device=torch.device('cuda')):
#         super(ConsistencyGraphModule, self).__init__()
#         self.in_feature = in_feature
#         self.out_feature = out_feature
#         self.device = device
#
#         self.linear1 = nn.Linear(self.in_feature, self.out_feature)
#         self.linear2 = nn.Linear(self.in_feature, self.out_feature)
#
#     def forward(self, x1, x2):
#         x_q = self.linear1(x1)
#         x_k = self.linear2(x2)
#         sim = cosine_sim(x_q, x_k, device=self.device)
#
#         return sim
#
# class Model1(nn.Module):
#     def __init__(self, n_views, in_dims, latent_dim, h_dim, device):
#         super(Model1, self).__init__()
#         self.n_views = n_views
#         self.in_dims = in_dims
#         self.latent_dim = latent_dim
#         self.h_dim = h_dim
#         self.device = device
#
#         self.encoders = []
#         self.decoders = []
#         for v in range(self.n_views):
#             self.encoders.append(Encoder(self.in_dims[v], self.latent_dim).to(self.device))
#             self.decoders.append(Decoder(self.in_dims[v], self.latent_dim).to(self.device))
#         self.encoders = nn.ModuleList(self.encoders)
#         self.decoders = nn.ModuleList(self.decoders)
#
#         self.shared_projector = nn.Linear(self.latent_dim, self.latent_dim)
#
#         self.high_level_projector = nn.Sequential(
#             nn.Linear(self.latent_dim, self.h_dim),
#         )
#
#         self.cross_attention_projector = CrossAttentionModule(self.latent_dim, self.latent_dim)
#
#     def forward(self, xs):
#         x_recs = []
#         zs = []
#         for v in range(self.n_views):
#             x = xs[v]
#             z = self.encoders[v](x)
#             # z = self.shared_projector(z)
#             z = F.normalize(z, dim=1)
#             x_rec = self.decoders[v](z)
#             zs.append(z)
#             x_recs.append(x_rec)
#         return  zs, x_recs
#
#     def HLP(self, xs, n_views = None):
#         if n_views is None:
#             n_views = self.n_views
#         hs = []
#         for v in range(n_views):
#             x = xs[v]
#             h = self.high_level_projector(x)
#             h = F.normalize(h, dim=1)
#             hs.append(h)
#         return hs
#
#     def CAG(self, xs, n_views = None):
#         if n_views is None:
#             n_views = self.n_views
#         z_atts = []
#         z_a = xs[0]
#         Atts = []
#         for b in range(1, n_views):
#             z_b = xs[b]
#             z_att, Att = self.cross_attention_projector(z_a, z_b)
#             z_att = F.normalize(z_att, dim=1)
#             z_atts.append(z_att)
#             Atts.append(Att)
#         return z_atts, Atts





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
                 dropout=0.0,
                 ):
        super(Autoencoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation
        self.batchnorm = batchnorm
        # self.dropout = dropout

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
        # 使用 * 解包，将列表中的每个层作为独立参数逐一传递给 nn.Sequential
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


class Projector(nn.Module):
    """Projector module."""
    def __init__(self, in_dim=128, hid_dim=256, out_dim=128, act='leakyrelu', dropout=0.1):
        super(Projector, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act = act
        self.dropout = dropout

        self.layer1 = nn.Linear(self.in_dim, self.hid_dim)
        self.layer2 = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer1(x)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)

        return x


class Model(nn.Module):
    """Model. View specific autoencoders.

    Parameters:
        - in_dim: a list of int. Dimensions of the input features view^1 and views^2.
        - hidden_dim: int. Dimension of the hidden layers in encoder/decoder.
        - emb_dim: int. Dimension of the latent embeddings.
        - p_dim: int. Dimension of the projector.
        - n_layers_e: positive int. Number of the encoder layers.
        - n_layers_d: positive int. Number of the decoder layers.
        - activation: activation method. Including "sigmoid", "tanh", "relu", "leakyrelu".
            default: "relu".
        - batchnorm: boolean. It provided whether to use the
            batchnorm in autoencoders.
        - dropout: dropout ratio if use dropout.
        - device: the device on which to train the model.
        """
    def __init__(self,
                 in_dim,
                 emb_dim,
                 hidden_dim=2000,
                 n_layers_e=4,
                 n_layers_d=4,
                 activation='relu',
                 batchnorm=True,
                 dropout=0.0,
                 device = torch.device('cuda')):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.device = device

        # View-specific autoencoders
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

        # self.projector1 = Projector(in_dim=self.emb_dim, hid_dim=256, out_dim=self.p_dim, dropout=self.dropout)
        # self.projector2 = Projector(in_dim=self.emb_dim, hid_dim=256, out_dim=self.p_dim, dropout=self.dropout)

        # self.shared_projector = nn.Linear(self.emb_dim, self.emb_dim)

        # self.high_level_projector = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.p_dim),
        # )
        self.high_level_projector1 = nn.Linear(self.emb_dim, self.emb_dim)
        # self.high_level_projector2 = nn.Linear(self.emb_dim, self.p_dim)

        # self.cross_attention_projector = CrossAttentionModule(self.emb_dim, self.emb_dim)

    def forward(self, X1, X2, is_eval=False):

        emb1 = self.autoencoder1.encode(X1)
        emb2 = self.autoencoder2.encode(X2)

        # emb1 = self.shared_projector(emb1)
        # emb2 = self.shared_projector(emb2)

        # emb1 = F.normalize(emb1, p=2, dim=-1)
        # emb2 = F.normalize(emb2, p=2, dim=-1)
        # emb1 = self.feature_standardize(emb1)
        # emb2 = self.feature_standardize(emb2)

        if is_eval:
            return emb1, emb2
        else:
            X1_hat = self.autoencoder1.decode(emb1)
            X2_hat = self.autoencoder2.decode(emb2)
            return emb1, emb2, X1_hat, X2_hat

    def high_level_project(self, emb1, emb2):
        H1 = self.high_level_projector1(emb1)
        H2 = self.high_level_projector1(emb2)

        H1 = F.normalize(H1, dim=1, p=2)
        H2 = F.normalize(H2, dim=1, p=2)

        return H1, H2

    # def CAM(self, emb1, emb2):
    #     att_score = self.cross_attention_projector(emb1, emb2)
    #     return att_score


    def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
        return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)

    def cross_view_consistency_loss(self, emb_aligned1, emb_aligned2, sim_aligned=None, n_neighbors=5):
        # graph_fu = self.aligned_fusion_graph(emb_aligned1.detach(), emb_aligned2.detach(), n_neighbors)
        if sim_aligned is None:
            sim_aligned = cosine_sim(emb_aligned1, emb_aligned2, device=self.device)
        # # target = graph_fu + torch.eye(graph_fu.size()[0], device=self.device)
        # idx = torch.ones(graph_fu.size()[0], device=self.device).bool()
        # graph_fu[idx, idx] = 1.0

        # # ||S - I||
        # eye_ = torch.eye(sim_aligned.size()[0], device=self.device)
        # loss = F.mse_loss(sim_aligned, eye_, reduction='mean')

        # ||diag(S) - 1||
        diag_ = torch.diag(sim_aligned)
        ones_ = torch.ones_like(diag_, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(diag_, ones_)

        # Covariance matching
        sim_matrix_1 = cosine_sim(emb_aligned1, emb_aligned1, device=self.device)
        sim_matrix_2 = cosine_sim(emb_aligned2, emb_aligned2, device=self.device)
        loss += F.mse_loss(sim_matrix_1, sim_matrix_2)

        return loss
        # return F.mse_loss(sim_aligned, graph_fu, reduction='mean')

    def cross_view_ncl(self, H1, H2, W, tau=1.0):
        # Lap = self.compute_laplacian(adj, self.device)
        sim_inter_view = cosine_sim(H1, H2, device=self.device)
        # x_sim_matrix = self.sim_matrix
        sim_inter_view = torch.exp(sim_inter_view / tau)

        # pos_sim = torch.diag(sim_inter_modal)
        # (w * Neu) / (Neu + Neg)
        loss_ = torch.sum(sim_inter_view * W, dim=-1) / (torch.sum(sim_inter_view, dim=-1))
        loss_inter_ncl = - torch.log(loss_).mean()

        # numerator = torch.sum(sim_inter_view * W, dim=-1)
        # denominator = torch.sum(sim_inter_view, dim=-1)
        # loss_ = numerator / denominator
        # loss_log = - torch.log(loss_)
        # loss_inter_ncl = loss_log.mean()

        return loss_inter_ncl

    def cl_wo_neutral_pairs(self, H1, H2, tau=1.0):
        sim_inter_view = cosine_sim(H1, H2, device=self.device)
        sim_inter_view = torch.exp(sim_inter_view / tau)

        # Pos / (Pos + Neg)
        # if flag is not None:
        #     # sim: N x N, use aligned and unaligned data
        #     pos_sim = sim_inter_view[flag, flag]
        #     loss_ = pos_sim / (torch.sum(sim_inter_view, dim=-1))
        #     loss_cl = - torch.log(loss_).mean()
        # else:
        #     # sim: Na x Na, only use aligned data
        pos_sim = torch.diag(sim_inter_view)
        loss_ = pos_sim / (torch.sum(sim_inter_view, dim=-1))
        loss_cl = - torch.log(loss_).mean()

        # numerator = torch.sum(sim_inter_view * W, dim=-1)
        # denominator = torch.sum(sim_inter_view, dim=-1)
        # loss_ = numerator / denominator
        # loss_log = - torch.log(loss_)
        # loss_inter_ncl = loss_log.mean()

        return loss_cl

    def get_weighted_adj(self, score, flag, aligned_score, k=1.0):
        """
        score: NxN; flag: N; aligned_score: N_a, the number of aligned samples.
        return weighted_adj: NxN
        """
        # alpha = 1.0
        diag_score = torch.diag(score)
        weights_diag = torch.zeros(diag_score.size(), device=self.device)
        # # Neutral pairs in diagonal
        # min_aligned_score = torch.min(aligned_score)
        # indices_neutral = diag_score >= min_aligned_score
        # weights_diag[indices_neutral] = diag_score[indices_neutral]
        # Positive pairs in diagonal
        mean_aligned_score = torch.mean(aligned_score)
        std_aligned_score = torch.std(aligned_score)
        # min_aligned_score = torch.min(aligned_score)
        # threshold = min_aligned_score
        # threshold = mean_aligned_score - std_aligned_score
        threshold = mean_aligned_score - k * std_aligned_score
        # threshold = mean_aligned_score
        threshold = max(threshold, 0.01)
        # threshold = 0.005
        indices_neutral = diag_score >= threshold
        weights_diag[indices_neutral] = diag_score[indices_neutral]
        # Positive pairs in diagonal (the aligned pairs)
        weights_diag[flag] = 1

        weighted_adj = score - torch.diag_embed(torch.diag(score))
        # Neutral pairs in non-diagonal
        weighted_adj = torch.where((weighted_adj >= threshold), weighted_adj, 0)

        weighted_adj = weighted_adj + torch.diag_embed(weights_diag)

        # weighted_adj = min_max_normalize(weighted_adj)

        # Make sure there is no row with all zeros.
        sum_row_ = torch.sum(weighted_adj, dim=-1)
        indices_zero_ = torch.where(sum_row_ == 0)
        if len(indices_zero_[0]) > 0:
            idices_max_ = torch.argmax(score[indices_zero_[0]], dim=-1)
            weighted_adj[indices_zero_[0], idices_max_] = 0.01

        return weighted_adj

    def aligned_fusion_graph(self, z1, z2, n_neighbors=3):
        graph1 = self.intra_sim_graph(z1, k=n_neighbors)
        graph2 = self.intra_sim_graph(z2, k=n_neighbors)
        graph_fu = graph1 + graph2
        graph_fu = torch.where(graph_fu > 1, 1, 0)

        return graph_fu.float()

    def intra_sim_graph(self, z, k=3):
        sim = cosine_sim(z, z, device=self.device)
        diag = torch.diag(sim)
        sim = sim - torch.diag_embed(diag)
        values, indices = torch.topk(sim, k=k, dim=-1, largest=True)
        adj = torch.zeros(sim.size(), dtype=torch.float32, device=self.device)
        adj = adj.scatter_(1, indices, 1)

        return adj

    def cross_corr_coef_loss(self, emb1, emb2, corr_coef_matrix=None, test_id=0):
        # Cross-view covariance
        if corr_coef_matrix is None:
            corr_coef_matrix = self.cross_corr_coef_matrix(emb1, emb2)
        diag_ = torch.diag(corr_coef_matrix)
        ones_ = torch.ones_like(diag_, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(diag_, ones_)

        # Covariance matching
        corr_coef_matrix_1 = self.cross_corr_coef_matrix(emb1, emb1)
        corr_coef_matrix_2 = self.cross_corr_coef_matrix(emb2, emb2)
        loss += F.mse_loss(corr_coef_matrix_1, corr_coef_matrix_2)

        # if test_id == 1:
        #     # Cross-view covariance
        #     if corr_coef_matrix is None:
        #         corr_coef_matrix = self.cross_corr_coef_matrix(emb1, emb2)
        #     diag_ = torch.diag(corr_coef_matrix)
        #     ones_ = torch.ones_like(diag_, dtype=torch.float32, device=self.device)
        #     loss = F.mse_loss(diag_, ones_)
        # elif test_id == 2:
        #     # Covariance matching
        #     corr_coef_matrix_1 = self.cross_corr_coef_matrix(emb1, emb1)
        #     corr_coef_matrix_2 = self.cross_corr_coef_matrix(emb2, emb2)
        #     loss = F.mse_loss(corr_coef_matrix_1, corr_coef_matrix_2)

        # graph_fu = self.aligned_fusion_graph(emb1.detach(), emb2.detach(), n_neighbors)
        # idx = torch.ones(graph_fu.size()[0], device=self.device).bool()
        # graph_fu[idx, idx] = 1.0
        # eye_ = torch.eye(corr_coef_matrix.size()[0], dtype=torch.float32, device=self.device)
        # loss = F.mse_loss(corr_coef_matrix, eye_)

        return loss

    def cross_corr_coef_matrix(self, emb1, emb2):
        emb1_std = self.feature_standardize(emb1)
        emb2_std = self.feature_standardize(emb2)
        N, D = emb1.size()
        eps = 1e-5
        eye_ = torch.eye(N, dtype=torch.float32, device=self.device)
        cov_matrix = torch.mm(emb1_std, emb2_std.t()) / (D - 1)  + (eps * eye_)

        return cov_matrix

    def feature_standardize(self, feature):
        feat_mean = torch.mean(feature.detach(), dim=-1).unsqueeze(-1)
        feat_std = torch.std(feature.detach(), dim=-1).unsqueeze(-1)
        ones_row_vec = torch.ones(feature.size(-1), dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_standardized = (feature - (torch.mm(feat_mean, ones_row_vec))).div(torch.mm(feat_std, ones_row_vec))

        return feat_standardized

    def cross_view_ncl2(self, H1, H2, W, tau=1.0):
        sim_inter_view = cosine_sim(H1, H2, device=self.device)
        sim_inter_view = torch.exp(sim_inter_view / tau)

        # pos_sim = torch.diag(sim_inter_modal)
        # (w * Neu) / (Neu + Neg)
        loss_ = torch.sum(sim_inter_view * W, dim=-1) / (torch.sum(sim_inter_view, dim=-1))
        loss_inter_ncl = - torch.log(loss_).mean()

        # numerator = torch.sum(sim_inter_view * W, dim=-1)
        # denominator = torch.sum(sim_inter_view, dim=-1)
        # loss_ = numerator / denominator
        # loss_log = - torch.log(loss_)
        # loss_inter_ncl = loss_log.mean()

        return loss_inter_ncl


    def inter_knn_graph(self, z1, z2, k=5):
        # adj = cosine_sim(z1, z2, device=self.device)
        corr_matrix = self.cross_corr_coef_matrix(z1, z2)
        corr_matrix = min_max_normalize(corr_matrix)        # scale to the range from 0 to 1

        values, indices = torch.topk(corr_matrix, k=k, dim=-1, largest=True)
        adj_inter = torch.zeros(corr_matrix.size(), dtype=torch.float32, device=self.device)
        adj_inter = adj_inter.scatter_(1, indices, 1)

        return adj_inter


    # def attention_adj(self, att_score, n_neighbors=3, device=torch.device('cuda')):
    #     values, indices = torch.topk(att_score, k=n_neighbors, dim=-1, largest=True)
    #     adj = torch.zeros(att_score.size(), dtype=torch.int64, device=device)
    #     adj = adj.scatter_(1, indices, 1)  # 在每行中按照列索引将值 scatter 到目标张量中
    #     # adj = adj + adj.t()
    #     # adj = torch.clamp(adj, max=1)
    #     return adj

    # def get_weight_adj(self, score, aligned_score):
    #     """
    #     score: NxN; aligned_score: N_a, the number of aligned samples.
    #     return weighted_adj: NxN
    #     """
    #     weighted_adj = score - torch.diag_embed(torch.diag(score))
    #     # Neutral pairs in non-diagonal
    #     mean_aligned_score = torch.mean(aligned_score)
    #     std_aligned_score = torch.std(aligned_score)
    #     threshold = mean_aligned_score - std_aligned_score
    #     weighted_adj = torch.where(weighted_adj >= threshold, weighted_adj, 0)
    #
    #     return weighted_adj


    # def compute_laplacian(self, adj, device):
    #     degree_arr = torch.sum(adj, dim=1)
    #     degree_arr = 1 / degree_arr.pow(0.5)
    #     degree_matrx = torch.diag(degree_arr)
    #     Lap = torch.eye(adj.size()[0]).to(device) - torch.mm(torch.mm(degree_matrx, adj), degree_matrx)
    #
    #     return Lap


    # ================================== 2025.02.27 ============================================
    # def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
    #     return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)
    #
    # def intra_modal_NCL(self, z_p1, z_p2, threshold=0.5, tau=1.0):
    #
    #     adj_intra = self.intra_sim_graph(z_p1, threshold=threshold)  # adj without self loops here.
    #     # x_sim_matrix = self.sim_matrix
    #     sim_intra_modal = cosine_sim(z_p1, z_p2, device=self.device)
    #     sim_intra_modal = torch.exp(sim_intra_modal / tau)
    #
    #     pos_sim = torch.diag(sim_intra_modal)
    #     # (w * Neu) / (Neu + Neg)
    #     loss_intra_ncl = (pos_sim + torch.sum(sim_intra_modal * adj_intra, dim=-1)) / (
    #         torch.sum(sim_intra_modal, dim=-1))
    #     loss_intra_ncl = - torch.log(loss_intra_ncl).mean()
    #
    #     return loss_intra_ncl
    #
    # def inter_modal_NCL(self, emb1, emb2, z1, z2, threshold=0.5, tau=1.0):
    #     self.adj_inter = self.inter_sim_graph(z1, z2, threshold=threshold)  # adj with self loops here.
    #     sim_inter_modal = cosine_sim(emb1, emb2, device=self.device)
    #     # x_sim_matrix = self.sim_matrix
    #     sim_inter_modal = torch.exp(sim_inter_modal / tau)
    #
    #     # pos_sim = torch.diag(sim_inter_modal)
    #     # (w * Neu) / (Neu + Neg)
    #     loss_inter_ncl = torch.sum(sim_inter_modal * self.adj_inter, dim=-1) / (torch.sum(sim_inter_modal, dim=-1))
    #     loss_inter_ncl = - torch.log(loss_inter_ncl).mean()
    #
    #     return loss_inter_ncl
    #
    # def intra_sim_graph(self, z, threshold=0.5):
    #     adj = cosine_sim(z, z, device=self.device)
    #     adj = min_max_normalize(adj)
    #     adj = torch.where(adj >= threshold, adj, 0)
    #     # pseudo_adj = pseudo_adj + sim_matrix
    #     # pseudo_adj = pseudo_adj + T_matrix
    #
    #     diag = torch.diag(adj)  # remove self loops
    #     adj = adj - torch.diag_embed(diag)
    #     # pseudo_adj = torch.clamp(pseudo_adj, max=1)
    #
    #     return adj
    #
    # def inter_sim_graph(self, z1, z2, threshold=0.5):
    #     adj = cosine_sim(z1, z2, device=self.device)
    #     adj = min_max_normalize(adj)  # scale to the range from 0 to 1
    #     adj = torch.where(adj >= threshold, adj, 0)
    #     # pseudo_adj = torch.clamp(pseudo_adj, max=1)
    #
    #     return adj
    # =========================================================================


class Model_Shared(nn.Module):
    """Model_Shared. Autoencoder with shared weight.

    Parameters:
        - in_dim: a list of int. Dimensions of the input features view^1 and views^2.
        - hidden_dim: int. Dimension of the hidden layers in encoder/decoder.
        - emb_dim: int. Dimension of the latent embeddings.
        - p_dim: int. Dimension of the projector.
        - n_layers_e: positive int. Number of the encoder layers.
        - n_layers_d: positive int. Number of the decoder layers.
        - activation: activation method. Including "sigmoid", "tanh", "relu", "leakyrelu".
            default: "relu".
        - batchnorm: boolean. It provided whether to use the
            batchnorm in autoencoders.
        - dropout: dropout ratio if use dropout.
        - device: the device on which to train the model.
    """
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 emb_dim,
                 p_dim,
                 n_layers_e,
                 n_layers_d,
                 activation='relu',
                 batchnorm=True,
                 dropout=0.0,
                 device=torch.device('cpu')):
        super(Model_Shared, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.p_dim = p_dim
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.device = device

        if self.n_layers_e == 1:
            # Only 1 layer
            self.encoder1 = nn.Linear(self.in_dim[0], self.emb_dim)
            self.encoder2 = nn.Linear(self.in_dim[1], self.emb_dim)
        elif self.n_layers_e > 1:
            # More than 1 layer
            encoder_input_layer1 = nn.Linear(self.in_dim[0], self.hidden_dim)
            encoder_input_layer2 = nn.Linear(self.in_dim[1], self.hidden_dim)
            encoder_layers = []
            for i in range(1, self.n_layers_e):
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
                if i == self.n_layers_e - 1:
                    encoder_layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
                else:
                    encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            self.encoder1 = [encoder_input_layer1] + encoder_layers
            self.encoder2 = [encoder_input_layer2] + encoder_layers
            self.encoder1 = nn.Sequential(*self.encoder1)
            self.encoder2 = nn.Sequential(*self.encoder2)

        decoder_layers = []
        if self.n_layers_d == 1:
            # Only 1 layer
            self.decoder1 = nn.Linear(self.emb_dim, self.in_dim[0])
            self.decoder2 = nn.Linear(self.emb_dim, self.in_dim[1])
        if self.n_layers_d > 1:
            for i in range(0, self.n_layers_d - 1):
                if i == 0:
                    decoder_layers.append(nn.Linear(self.emb_dim, self.hidden_dim))
                else:
                    decoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
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

            decoder_output_layer1 = nn.Linear(self.hidden_dim, self.in_dim[0])
            decoder_output_layer2 = nn.Linear(self.hidden_dim, self.in_dim[1])
            self.decoder1 = decoder_layers + [decoder_output_layer1]
            self.decoder2 = decoder_layers + [decoder_output_layer2]
            self.decoder1 = nn.Sequential(*self.decoder1)
            self.decoder2 = nn.Sequential(*self.decoder2)

        self.projector1 = Projector(in_dim=self.emb_dim, out_dim=self.p_dim, dropout=self.dropout)
        self.projector2 = Projector(in_dim=self.emb_dim, out_dim=self.p_dim, dropout=self.dropout)

    def forward(self, X1, X2, is_eval=False):

        emb1 = self.encoder1(X1)
        emb2 = self.encoder2(X2)

        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        if is_eval:
            return emb1, emb2
        else:
            X1_hat = self.decoder1(emb1)
            X2_hat = self.decoder2(emb2)
            return emb1, emb2, X1_hat, X2_hat

    def project(self, emb1, emb2):
        Z1_p1 = self.projector1(emb1)
        Z2_p1 = self.projector1(emb2)
        Z1_p2 = self.projector2(add_gaussian_noise(emb1))
        Z2_p2 = self.projector2(add_gaussian_noise(emb2))

        Z1_p1 = F.normalize(Z1_p1, dim=1, p=2)
        Z2_p1 = F.normalize(Z2_p1, dim=1, p=2)
        Z1_p2 = F.normalize(Z1_p2, dim=1, p=2)
        Z2_p2 = F.normalize(Z2_p2, dim=1, p=2)

        # return Z1_p1, Z2_p1
        return Z1_p1, Z2_p1, Z1_p2, Z2_p2

    def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
        return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)

    def intra_modal_NCL(self, z_p1, z_p2,  threshold=0.5, tau=1.0):

        adj_intra = self.intra_graph(z_p1, threshold=threshold)  # adj without self loops here.
        # x_sim_matrix = self.sim_matrix
        sim_intra_modal = cosine_sim(z_p1, z_p2, device=self.device)
        sim_intra_modal = torch.exp(sim_intra_modal / tau)

        # diag = torch.diag(sim_intra_modal)
        pos_sim = torch.diag(sim_intra_modal)
        # (w * Neu) / (Neu + Neg)
        loss_intra_ncl = (pos_sim + torch.sum(sim_intra_modal * adj_intra, dim=-1)) / (
            torch.sum(sim_intra_modal, dim=-1))
        loss_intra_ncl = - torch.log(loss_intra_ncl).mean()

        return loss_intra_ncl

    def inter_modal_NCL(self, emb1, emb2, z1, z2, threshold=0.5, tau=1.0):
        # self.adj_inter = self.inter_sim_graph(z1, z2, threshold=threshold)  # adj with self loops here.
        self.inter_graph(z1, z2, emb1, emb2, threshold=threshold)           # adj with self loops here.
        sim_inter_modal = cosine_sim(emb1, emb2, device=self.device)
        # x_sim_matrix = self.sim_matrix
        sim_inter_modal = torch.exp(sim_inter_modal / tau)

        diag_adj = torch.diag(self.adj_inter)
        diag_sim = torch.diag(sim_inter_modal) * diag_adj
        # (w * Neu) / (Neu + Neg)
        loss_inter_ncl = (diag_sim + torch.sum(sim_inter_modal * (self.adj_inter - torch.diag_embed(diag_adj)), dim=-1)) / (torch.sum(sim_inter_modal, dim=-1))
        # loss_inter_ncl = torch.sum(sim_inter_modal * self.adj_inter, dim=-1) / (torch.sum(sim_inter_modal, dim=-1))
        loss_inter_ncl = - torch.log(loss_inter_ncl).mean()

        # emb2_hat = torch.matmul(self.adj_inter, emb2)
        # emb2_hat = F.normalize(emb2_hat, p=2, dim=-1)
        # loss_inter_ncl += F.mse_loss(emb1, emb2_hat)

        return loss_inter_ncl

    def intra_graph(self, z, threshold=0.5):
        adj = cosine_sim(z, z, device=self.device)
        adj = min_max_normalize(adj)
        adj = torch.where(adj >= threshold, adj, 0)
        # pseudo_adj = pseudo_adj + sim_matrix
        # pseudo_adj = pseudo_adj + T_matrix

        diag = torch.diag(adj)  # remove self loops
        adj = adj - torch.diag_embed(diag)
        # pseudo_adj = torch.clamp(pseudo_adj, max=1)

        return adj

    def inter_graph(self, z1, z2, emb1, emb2, threshold=0.5):
        # adj_0 = self.adj_inter.clone()
        adj = cosine_sim(z1, z2, device=self.device)
        adj = min_max_normalize(adj)        # scale to the range from 0 to 1
        # adj_max = torch.max(adj, dim=-1)
        # adj_min = torch.min(adj, dim=-1)
        diag = torch.diag(adj)
        omega0 = (diag > 0).int().sum()
        omega1 = (diag == 1).int().sum()
        omega = (diag > threshold).int().sum()
        # adj = adj - torch.diag_embed(diag)
        self.adj_inter = torch.where(adj >= threshold, adj, 0)
        # omega1 = (torch.diag(self.adj_inter)>0).int().sum()
        self.adj_inter = self.adj_inter + torch.diag_embed(diag)    # remain self loop
        # adj = torch.where(adj >= threshold, adj, 0)
        # # pseudo_adj = torch.clamp(pseudo_adj, max=1)
        #
        # emb2_hat = torch.matmul(adj, emb2)
        # emb2_hat = F.normalize(emb2_hat, p=2, dim=-1)
        # omega = (adj > 0).int()
        # omega = torch.sum(adj, dim=-1) / torch.sum(omega, dim=-1)
        # omega = torch.unsqueeze(omega, dim=-1)
        # self.adj_inter = adj + torch.mul(omega**2, cosine_sim(emb1, torch.matmul(adj, emb2_hat), device=self.device))
        # # self.adj_inter = torch.clamp(adj, max=1)

    def compute_cosine_sim(self, x1, x2):
        self.sim_matrix = cosine_sim(x1, x2, device=self.device)
        return self.sim_matrix

# # ====================================== 0914 ===================================
#     def intra_modal_NCL(self, z, threshold=0.8, tau=1.0):
#         sim_intra_modal = cosine_sim(z, z, device=self.device)
#         adj_intra = self.intra_sim_graph(sim_intra_modal, threshold=threshold)          # adj without self loops here.
#         # x_sim_matrix = self.sim_matrix
#         sim_intra_modal = torch.exp(sim_intra_modal / tau)
#
#         diag = torch.diag(sim_intra_modal)
#         sim_intra_modal = sim_intra_modal - torch.diag_embed(diag)
#         # (w * Neu) / (Neu + Neg)
#         loss_intra_ncl = torch.sum(sim_intra_modal * adj_intra, dim=-1) / (torch.sum(sim_intra_modal, dim=-1))
#         loss_intra_ncl = - torch.log(loss_intra_ncl).mean()
#
#         return loss_intra_ncl
#
#     def inter_modal_NCL(self, z1, z2, threshold=0.8, tau=1.0):
#         sim_inter_modal = cosine_sim(z1, z2, device=self.device)
#         self.adj_inter = self.inter_sim_graph(sim_inter_modal, threshold=threshold)          # adj with self loops here.
#         # x_sim_matrix = self.sim_matrix
#         sim_inter_modal = torch.exp(sim_inter_modal / tau)
#
#         # pos_sim = torch.diag(sim_inter_modal)
#         # (w * Neu) / (Neu + Neg)
#         loss_inter_ncl = torch.sum(sim_inter_modal * self.adj_inter, dim=-1) / (torch.sum(sim_inter_modal, dim=-1))
#         loss_inter_ncl = - torch.log(loss_inter_ncl).mean()
#
#         return loss_inter_ncl
#
#
#     def compute_cosine_sim(self, x1, x2):
#         self.sim_matrix = cosine_sim(x1, x2, device=self.device)
#         return self.sim_matrix
#
#     def intra_sim_graph(self, sim_matrix, threshold=0.8):
#         # s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
#         # s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
#         # adj = (sim_matrix - s_min) / (s_max - s_min)
#
#         adj = min_max_normalize(sim_matrix)
#
#         adj = torch.where(adj >= threshold, adj, 0)
#         # pseudo_adj = pseudo_adj + sim_matrix
#         # pseudo_adj = pseudo_adj + T_matrix
#
#         diag = torch.diag(adj)                  # remove self loops
#         adj = adj - torch.diag_embed(diag)
#
#         # pseudo_adj = torch.clamp(pseudo_adj, max=1)
#
#         return adj
#
#     def inter_sim_graph(self, sim_matrix, threshold=0.8):
#         # s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
#         # s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
#         # adj = (sim_matrix - s_min) / (s_max - s_min)
#
#         adj = min_max_normalize(sim_matrix)
#
#         adj = torch.where(adj >= threshold, adj, 0)
#         # pseudo_adj = pseudo_adj + sim_matrix
#         # pseudo_adj = pseudo_adj + T_matrix
#
#         # pseudo_adj = torch.clamp(pseudo_adj, max=1)
#
#         return adj
# # ====================================== 0914 ===================================


# class Model_Shared0(nn.Module):
#     """Model_Shared. Autoencoder with shared weight.
#
#     Parameters:
#         - in_dim: a list of int. Dimensions of the input features view^1 and views^2.
#         - hidden_dim: int. Dimension of the hidden layers in encoder/decoder.
#         - emb_dim: int. Dimension of the latent embeddings.
#         - p_dim: int. Dimension of the projector.
#         - n_layers_e: positive int. Number of the encoder layers.
#         - n_layers_d: positive int. Number of the decoder layers.
#         - activation: activation method. Including "sigmoid", "tanh", "relu", "leakyrelu".
#             default: "relu".
#         - batchnorm: boolean. It provided whether to use the
#             batchnorm in autoencoders.
#         - dropout: dropout ratio if use dropout.
#         - device: the device on which to train the model.
#     """
#     def __init__(self,
#                  in_dim,
#                  hidden_dim,
#                  emb_dim,
#                  p_dim,
#                  n_layers_e,
#                  n_layers_d,
#                  activation='relu',
#                  batchnorm=True,
#                  dropout=0.0,
#                  device=torch.device('cpu')):
#         super(Model_Shared0, self).__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.emb_dim = emb_dim
#         self.p_dim = p_dim
#         self.n_layers_e = n_layers_e
#         self.n_layers_d = n_layers_d
#         self.activation = activation
#         self.batchnorm = batchnorm
#         self.dropout = dropout
#         self.device = device
#
#         if self.n_layers_e == 1:
#             # Only 1 layer
#             self.encoder1 = nn.Linear(self.in_dim[0], self.emb_dim)
#             self.encoder2 = nn.Linear(self.in_dim[1], self.emb_dim)
#         elif self.n_layers_e > 1:
#             # More than 1 layer
#             encoder_input_layer1 = nn.Linear(self.in_dim[0], self.hidden_dim)
#             encoder_input_layer2 = nn.Linear(self.in_dim[1], self.hidden_dim)
#             encoder_layers = []
#             for i in range(1, self.n_layers_e):
#                 if self.batchnorm:
#                     encoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
#                 if self.activation == 'sigmoid':
#                     encoder_layers.append(nn.Sigmoid())
#                 elif self.activation == 'relu':
#                     encoder_layers.append(nn.ReLU())
#                 elif self.activation == 'leakyrelu':
#                     encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 elif self.activation == 'tanh':
#                     encoder_layers.append(nn.Tanh())
#                 elif self.activation == 'none':
#                     encoder_layers.append(nn.Identity())
#                 else:
#                     raise ValueError(f"Unknown activation type {self.activation}")
#                 if i == self.n_layers_e - 1:
#                     encoder_layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
#                 else:
#                     encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
#
#             self.encoder1 = [encoder_input_layer1] + encoder_layers
#             self.encoder2 = [encoder_input_layer2] + encoder_layers
#             self.encoder1 = nn.Sequential(*self.encoder1)
#             self.encoder2 = nn.Sequential(*self.encoder2)
#
#         if self.n_layers_d == 1:
#             # Only 1 layer
#             self.decoder1 = nn.Linear(self.emb_dim, self.in_dim[0])
#             self.decoder2 = nn.Linear(self.emb_dim, self.in_dim[1])
#         elif self.n_layers_d > 1:
#             decoder_layers = []
#             for i in range(0, self.n_layers_d - 1):
#                 if i == 0:
#                     decoder_layers.append(nn.Linear(self.emb_dim, self.hidden_dim))
#                 else:
#                     decoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
#                 if self.batchnorm:
#                     decoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
#                 if self.activation == 'sigmoid':
#                     decoder_layers.append(nn.Sigmoid())
#                 elif self.activation == 'relu':
#                     decoder_layers.append(nn.ReLU())
#                 elif self.activation == 'leakyrelu':
#                     decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 elif self.activation == 'tanh':
#                     decoder_layers.append(nn.Tanh())
#                 elif self.activation == 'none':
#                     decoder_layers.append(nn.Identity())
#                 else:
#                     raise ValueError(f"Unknown activation type {self._activation}")
#
#             decoder_output_layer1 = nn.Linear(self.hidden_dim, self.in_dim[0])
#             decoder_output_layer2 = nn.Linear(self.hidden_dim, self.in_dim[1])
#             self.decoder1 = decoder_layers + [decoder_output_layer1]
#             self.decoder2 = decoder_layers + [decoder_output_layer2]
#             self.decoder1 = nn.Sequential(*self.decoder1)
#             self.decoder2 = nn.Sequential(*self.decoder2)
#
#         self.projector1 = Projector(in_dim=self.emb_dim, out_dim=self.p_dim, dropout=self.dropout)
#         self.projector2 = Projector(in_dim=self.emb_dim, out_dim=self.p_dim, dropout=self.dropout)
#
#     def forward(self, X1, X2, is_eval=False):
#
#         emb1 = self.encoder1(X1)
#         emb2 = self.encoder2(X2)
#
#         emb1 = F.normalize(emb1, p=2, dim=-1)
#         emb2 = F.normalize(emb2, p=2, dim=-1)
#
#         if is_eval:
#             return emb1, emb2
#         else:
#             X1_hat = self.decoder1(emb1)
#             X2_hat = self.decoder2(emb2)
#             return emb1, emb2, X1_hat, X2_hat
#
#     def project(self, emb1, emb2):
#         Z1_p1 = self.projector1(emb1)
#         Z2_p1 = self.projector1(emb2)
#         Z1_p2 = self.projector2(emb1)
#         Z2_p2 = self.projector2(emb2)
#
#         return Z1_p1, Z2_p1, Z1_p2, Z2_p2
#
#     def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
#         return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)
#
#     def intra_modal_NCL(self, z, threshold=0.8, tau=1.0):
#         sim_intra_modal = cosine_sim(z, z, device=self.device)
#         self.adj_intra = self.intra_sim_graph(sim_intra_modal, threshold=threshold)  # adj without self loops here.
#         # x_sim_matrix = self.sim_matrix
#         sim_intra_modal = torch.exp(sim_intra_modal / tau)
#
#         # pos_sim = torch.diag(sim_intra_modal)
#         # # (Pos + w * Neu) / (Neu + Neg)
#         # loss_intra_ncl = (pos_sim + torch.sum(sim_intra_modal * self.adj_intra, dim=-1)) / (
#         #     torch.sum(sim_intra_modal, dim=-1))
#         # loss_intra_ncl = - torch.log(loss_intra_ncl).mean()
#
#         diag = torch.diag(sim_intra_modal)
#         # sim_intra_modal = sim_intra_modal - torch.diag_embed(diag)              # Bug
#         # (w * Neu) / (Neu + Neg)
#         loss_intra_ncl = torch.sum(sim_intra_modal * self.adj_intra, dim=-1) / (
#             torch.sum(sim_intra_modal, dim=-1))
#         loss_intra_ncl = - torch.log(loss_intra_ncl).mean()
#
#         return loss_intra_ncl
#
#     def inter_modal_NCL(self, z1, z2, threshold=0.8, tau=1.0):
#         sim_inter_modal = cosine_sim(z1, z2, device=self.device)
#         self.adj_inter = self.inter_sim_graph(sim_inter_modal, threshold=threshold)  # adj with self loops here.
#         # x_sim_matrix = self.sim_matrix
#         sim_inter_modal = torch.exp(sim_inter_modal / tau)
#
#         # pos_sim = torch.diag(sim_inter_modal)
#         # (w * Neu) / (Neu + Neg)
#         loss_inter_ncl = torch.sum(sim_inter_modal * self.adj_inter, dim=-1) / (torch.sum(sim_inter_modal, dim=-1))
#         loss_inter_ncl = - torch.log(loss_inter_ncl).mean()
#
#         return loss_inter_ncl
#
#     def compute_cosine_sim(self, x1, x2):
#         self.sim_matrix = cosine_sim(x1, x2, device=self.device)
#         return self.sim_matrix
#
#     def intra_sim_graph(self, sim_matrix, threshold=0.8):
#         # s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
#         # s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
#         # adj = (sim_matrix - s_min) / (s_max - s_min)
#
#         adj = min_max_normalize(sim_matrix)
#
#         adj = torch.where(adj >= threshold, adj, 0)
#         # pseudo_adj = pseudo_adj + sim_matrix
#         # pseudo_adj = pseudo_adj + T_matrix
#
#         diag = torch.diag(adj)  # remove self loops
#         adj = adj - torch.diag_embed(diag)
#
#         # pseudo_adj = torch.clamp(pseudo_adj, max=1)
#
#         return adj
#
#     def inter_sim_graph(self, sim_matrix, threshold=0.8):
#         # s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
#         # s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
#         # adj = (sim_matrix - s_min) / (s_max - s_min)
#
#         adj = min_max_normalize(sim_matrix)
#
#         adj = torch.where(adj >= threshold, adj, 0)
#         # pseudo_adj = pseudo_adj + sim_matrix
#         # pseudo_adj = pseudo_adj + T_matrix
#
#         # pseudo_adj = torch.clamp(pseudo_adj, max=1)
#
#         return adj





# class Autoencoder0(nn.Module):
#     """AutoEncoder module that projects features to latent space."""
#     def __init__(self,
#                  encoder_dim,
#                  activation='relu',
#                  batchnorm=True):
#         """Constructor.
#
#         Args:
#           encoder_dim: Should be a list of ints, hidden sizes of
#             encoder network, the last element is the size of the latent representation.
#           activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
#             simply choose relu.
#           batchnorm: if provided should be a bool type. It provided whether to use the
#             batchnorm in autoencoders.
#         """
#         super(Autoencoder0, self).__init__()
#
#         self._dim = len(encoder_dim) - 1
#         self._activation = activation
#         self._batchnorm = batchnorm
#
#         encoder_layers = []
#         for i in range(self._dim):
#             encoder_layers.append(
#                 nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
#             if i < self._dim - 1:
#                 if self._batchnorm:
#                     encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
#                 if self._activation == 'sigmoid':
#                     encoder_layers.append(nn.Sigmoid())
#                 elif self._activation == 'relu':
#                     encoder_layers.append(nn.ReLU())
#                 elif self._activation == 'leakyrelu':
#                     encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 elif self._activation == 'tanh':
#                     encoder_layers.append(nn.Tanh())
#                 elif self._activation == 'none':
#                     encoder_layers.append(nn.Identity())
#                 else:
#                     raise ValueError('Unknown activation type %s' % self._activation)
#         encoder_layers.append(nn.Softmax(dim=1))
#         self._encoder = nn.Sequential(*encoder_layers)
#
#         decoder_dim = [i for i in reversed(encoder_dim)]
#         decoder_layers = []
#         for i in range(self._dim):
#             decoder_layers.append(
#                 nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
#             if self._batchnorm:
#                 decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
#             if self._activation == 'sigmoid':
#                 decoder_layers.append(nn.Sigmoid())
#             elif self._activation == 'relu':
#                 decoder_layers.append(nn.ReLU())
#             elif self._activation == 'leakyrelu':
#                 decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#             elif self._activation == 'tanh':
#                 decoder_layers.append(nn.Tanh())
#             elif self._activation == 'none':
#                 decoder_layers.append(nn.Identity())
#             else:
#                 raise ValueError('Unknown activation type %s' % self._activation)
#         self._decoder = nn.Sequential(*decoder_layers)
#
#     def encoder(self, x):
#         """Encode sample features.
#
#             Args:
#               x: [num, feat_dim] float tensor.
#
#             Returns:
#               latent: [n_nodes, latent_dim] float tensor, representation Z.
#         """
#         latent = self._encoder(x)
#         return latent
#
#     def decoder(self, latent):
#         """Decode sample features.
#
#             Args:
#               latent: [num, latent_dim] float tensor, representation Z.
#
#             Returns:
#               x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
#         """
#         x_hat = self._decoder(latent)
#         return x_hat
#
#     def forward(self, x):
#         """Pass through autoencoder.
#
#             Args:
#               x: [num, feat_dim] float tensor.
#
#             Returns:
#               latent: [num, latent_dim] float tensor, representation Z.
#               x_hat:  [num, feat_dim] float tensor, reconstruction x.
#         """
#         latent = self.encoder(x)
#         x_hat = self.decoder(latent)
#         return x_hat, latent
#
#
# class Generator(nn.Module):
#     """Dual prediction module that projects features from corresponding latent space."""
#
#     def __init__(self, feature_dim, activation='relu', batchnorm=True):
#         """Constructor.
#
#         Args:
#           prediction_dim: Should be a list of ints, hidden sizes of
#             prediction network, the last element is the size of the latent representation of autoencoder.
#           activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
#             simply choose relu.
#           batchnorm: if provided should be a bool type. It provided whether to use the
#             batchnorm in autoencoders.
#         """
#         super(Generator, self).__init__()
#
#         self._depth = len(feature_dim) - 1
#         self._activation = activation
#         self._prediction_dim = feature_dim
#
#         encoder_layers = []
#         for i in range(self._depth):
#             encoder_layers.append(
#                 nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
#             if batchnorm:
#                 encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
#             if self._activation == 'sigmoid':
#                 encoder_layers.append(nn.Sigmoid())
#             elif self._activation == 'leakyrelu':
#                 encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#             elif self._activation == 'tanh':
#                 encoder_layers.append(nn.Tanh())
#             elif self._activation == 'relu':
#                 encoder_layers.append(nn.ReLU())
#             else:
#                 raise ValueError('Unknown activation type %s' % self._activation)
#         self._encoder = nn.Sequential(*encoder_layers)
#
#         decoder_layers = []
#         for i in range(self._depth, 0, -1):
#             decoder_layers.append(
#                 nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
#             if i > 1:
#                 if batchnorm:
#                     decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
#                 if self._activation == 'sigmoid':
#                     decoder_layers.append(nn.Sigmoid())
#                 elif self._activation == 'leakyrelu':
#                     decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 elif self._activation == 'tanh':
#                     decoder_layers.append(nn.Tanh())
#                 elif self._activation == 'relu':
#                     decoder_layers.append(nn.ReLU())
#                 else:
#                     raise ValueError('Unknown activation type %s' % self._activation)
#         decoder_layers.append(nn.Softmax(dim=1))
#         self._decoder = nn.Sequential(*decoder_layers)
#
#     def forward(self, x):
#         """Data recovery by prediction.
#
#             Args:
#               x: [num, feat_dim] float tensor.
#
#             Returns:
#               latent: [num, latent_dim] float tensor.
#               output:  [num, feat_dim] float tensor, recovered data.
#         """
#         latent = self._encoder(x)
#         output = self._decoder(latent)
#         return output, latent
#
# class Network(nn.Module):
#     def __init__(self, config):
#         super(Network, self).__init__()
#         if config['Autoencoder']['arch1'][-1] != config['Autoencoder']['arch2'][-1]:
#             raise ValueError('Inconsistent latent dim!')
#
#         self._latent_dim = config['Autoencoder']['arch1'][-1]
#         self._num_sample = config['training']['num_sample']
#         self._num_classes = config['training']['num_classes']
#         self._dims_view1 = [self._latent_dim] + config['Prediction']['arch1']
#         self._dims_view2 = [self._latent_dim] + config['Prediction']['arch2']
#
#         # View-specific autoencoders
#         self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
#                                         config['Autoencoder']['batchnorm'])
#         self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
#                                         config['Autoencoder']['batchnorm'])
#         self.generator1 = Generator(self._dims_view1)
#         self.generator2 = Generator(self._dims_view2)
#
#     def forward(self, x1_train, x2_train):
#         z1 = self.autoencoder1.encoder(x1_train)
#         z2 = self.autoencoder2.encoder(x2_train)
#         z1_hat, _ = self.generator1(z1.detach())
#         z2_hat, _ = self.generator2(z2.detach())
#         x1_recon = self.autoencoder1.decoder(z1)
#         x2_recon = self.autoencoder2.decoder(z2)
#         return x1_recon, x2_recon, z1, z2, z1_hat, z2_hat
#
#     def evaluation(self, logger, x1_train, x2_train, Y_list):
#         with torch.no_grad():
#             self.autoencoder1.eval(), self.autoencoder2.eval()
#             self.generator1.eval(), self.generator2.eval()
#             z1 = self.autoencoder1.encoder(x1_train)
#             z2 = self.autoencoder2.encoder(x2_train)
#             z1_hat, _ = self.generator1(z1)
#             z2_hat, _ = self.generator2(z2)
#             latent_fusion = torch.cat([z2_hat, z1_hat], dim=1).cpu().numpy()
#             scores = evaluation.clustering([latent_fusion], Y_list[0])
#             logger.info("\033[2;29m" + '     ===>pretrain ' + str(scores) + "\033[0m")
#             self.autoencoder1.train(), self.autoencoder2.train()
#             self.generator1.train(), self.generator2.train()
#         return scores
#
# class GOT(nn.Module):
#     def __init__(self, nodes, tau, it):
#         super(GOT, self).__init__()
#         self._nodes = nodes
#         self._tau = tau
#         self._it = it
#         self.mean = nn.Parameter(torch.rand((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)
#         self.std = nn.Parameter(10 * torch.ones((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)
#
#     def init_param(self, similarity):
#         self.mean.data = similarity
#
#     def doubly_stochastic(self, P):
#         """Uses logsumexp for numerical stability."""
#         A = P / self._tau
#         for i in range(self._it):
#             A = A - A.logsumexp(dim=1, keepdim=True)
#             A = A - A.logsumexp(dim=0, keepdim=True)
#         return torch.exp(A)
#
#     def forward(self, eps):
#         P_noisy = self.mean + self.std * eps
#         DS = self.doubly_stochastic(P_noisy)
#         return DS
#
#     def loss_got(self, g1, g2, DS, params):
#         [C1_tilde, C2_tilde] = params
#         loss_c = torch.trace(g1) + torch.trace(DS @ g2 @ torch.transpose(DS, 0, 1))
#         # svd version
#         u, sigma, v = torch.svd(C2_tilde @ torch.transpose(DS, 0, 1) @ C1_tilde)
#         loss = loss_c - 2 * torch.sum(sigma)
#         return loss
#
# class OTGM(nn.Module):
#     def __init__(self, config):
#         super(OTGM, self).__init__()
#         if config['Autoencoder']['arch1'][-1] != config['Autoencoder']['arch2'][-1]:
#             raise ValueError('Inconsistent latent dim!')
#         self._num_sample = config['training']['num_sample']
#         self._num_aligned = int(self._num_sample * config['training']['aligned_ratio'])
#         self._num_mis_aligned = self._num_sample - self._num_aligned
#
#         self.network = Network(config)
#         self.got = GOT(self._num_mis_aligned, config['training']['got']['tau'], config['training']['got']['it'])
#
#     def forward(self, x1_train, x2_train):
#         return self.network(x1_train, x2_train)



