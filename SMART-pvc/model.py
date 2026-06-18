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
                 activation='leakyrelu',
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
                 activation='leakyrelu',  # sigmoid, relu, leakyrelu, tanh, none
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

    def forward(self, X1, X2, is_eval=False):
        emb1 = self.autoencoder1.encode(X1)
        emb2 = self.autoencoder2.encode(X2)

        if is_eval:
            return emb1, emb2
        else:
            X1_hat = self.autoencoder1.decode(emb1)
            X2_hat = self.autoencoder2.decode(emb2)
            return emb1, emb2, X1_hat, X2_hat

    # 
    def high_level_project(self, emb1, emb2):
        H1 = self.high_level_projector1(emb1)
        H2 = self.high_level_projector1(emb2)

        H1 = F.normalize(H1, dim=1, p=2)
        H2 = F.normalize(H2, dim=1, p=2)

        return H1, H2


    # ====================== 1. 重建损失
    def reconstruction_loss(self, X1, X2, X1_hat, X2_hat):
        return F.mse_loss(X1_hat, X1) + F.mse_loss(X2_hat, X2)

    # ====================== 2. 视图分布对齐损失
    def VDA_loss(self, emb1, emb2, corr_coef_matrix=None):
        '''View Distribution Alignment'''
        # Covariance Matching Alignment
        if corr_coef_matrix is None:
            corr_coef_matrix = self.cross_corr_coef_matrix(emb1, emb2)
        diag_ = torch.diag(corr_coef_matrix)
        ones_ = torch.ones_like(diag_, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(diag_, ones_)

        # Cross-view Feature Alignment
        corr_coef_matrix_1 = self.cross_corr_coef_matrix(emb1, emb1)
        corr_coef_matrix_2 = self.cross_corr_coef_matrix(emb2, emb2)
        loss += F.mse_loss(corr_coef_matrix_1, corr_coef_matrix_2)

        return loss

    def cross_corr_coef_matrix(self, emb1, emb2):
        emb1_std = self.feature_standardize(emb1)
        emb2_std = self.feature_standardize(emb2)
        N, D = emb1.size()
        eps = 1e-5
        eye_ = torch.eye(N, dtype=torch.float32, device=self.device)
        cov_matrix = torch.mm(emb1_std, emb2_std.t()) / (D - 1)  + (eps * eye_)

        return cov_matrix


    # ======================  3. 语义匹配对比学习损失
    def SMC_loss(self, H1, H2, W, tau=1.0):
        '''Semantic Matching Contrastive Learning'''
        sim_inter_view = cosine_sim(H1, H2, device=self.device)
        sim_inter_view = torch.exp(sim_inter_view / tau)
        loss_ = torch.sum(sim_inter_view * W, dim=-1) / (torch.sum(sim_inter_view, dim=-1))
        loss_inter_smc = - torch.log(loss_).mean()

        return loss_inter_smc

    def get_semantic_graph(self, score, flag, aligned_score):
        """
        score: NxN; flag: N; aligned_score: N_a, the number of aligned samples.
        return weighted_adj: NxN
        """
        diag_score = torch.diag(score)
        weights_diag = torch.zeros(diag_score.size(), device=self.device)
        mean_aligned_score = torch.mean(aligned_score)
        std_aligned_score = torch.std(aligned_score)
        threshold = mean_aligned_score - std_aligned_score
        threshold = max(threshold, 0.01)
        indices_neutral = diag_score >= threshold
        weights_diag[indices_neutral] = diag_score[indices_neutral]
        # Positive pairs in diagonal (the aligned pairs)
        weights_diag[flag] = 1

        weighted_adj = score - torch.diag_embed(torch.diag(score))
        # Semantic pairs in non-diagonal
        weighted_adj = torch.where((weighted_adj >= threshold), weighted_adj, 0)
        weighted_adj = weighted_adj + torch.diag_embed(weights_diag)

        # Make sure there is no row with all zeros.
        sum_row_ = torch.sum(weighted_adj, dim=-1)
        indices_zero_ = torch.where(sum_row_ == 0)
        if len(indices_zero_[0]) > 0:
            idices_max_ = torch.argmax(score[indices_zero_[0]], dim=-1)
            weighted_adj[indices_zero_[0], idices_max_] = 0.01

        return weighted_adj


    def feature_standardize(self, feature):
        feat_mean = torch.mean(feature.detach(), dim=-1).unsqueeze(-1)
        feat_std = torch.std(feature.detach(), dim=-1).unsqueeze(-1)
        ones_row_vec = torch.ones(feature.size(-1), dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_standardized = (feature - (torch.mm(feat_mean, ones_row_vec))).div(torch.mm(feat_std, ones_row_vec))

        return feat_standardized
