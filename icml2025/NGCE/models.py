import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import cal_homo_ratio
from sklearn.metrics.pairwise import cosine_similarity
# from layers import GCNConv_dense, GCNConv_dgl
from utils import *

class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, order):
        super(GraphEncoder, self).__init__()
        self.LatentMap = LatentMappingLayer(input_dim, hidden_dim, output_dim, num_layers=2)
        self.order = order

    def forward(self, x, adj):

        adj = F.normalize(adj, p=2, dim=1)
        # print(adj)
        z = self.message_passing_global(x, adj, self.order)
        # z = self.LatentMap(z, dropout=False)    
        return z

    def message_passing_global(self, x, adj, order):
        h = x
        for i in range(order):
            h = torch.matmul(adj, h) + (1 * x)
        return h

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class GNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(GNN, self).__init__()
        self.gnn = GraphEncoder(feat_dim, hidden_dim, latent_dim)
        self.dec = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, adj, order):
        z = self.gnn(x, adj, order)
        z_norm = F.normalize(z, p=2, dim=1)
        a_pred = torch.sigmoid(torch.mm(z_norm, z_norm.t()))
        x_pred = torch.sigmoid(self.dec(F.relu(z), dropout=False))
        return z_norm, a_pred, x_pred




class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        # a_pred = torch.sigmoid(torch.mm(z, z.t()))
        return x_pred, z_norm



# masked X
class Masked_features(nn.Module):    
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(Masked_features, self).__init__()

    def forward(self, x, noise_mode):
        if noise_mode == 0:
            noise_X = self.get_random_mask(x, severity=0.2)   # wisconsin==0.3  cornell==0.2
        elif noise_mode == 1:
            noise_X = self.add_gaussian_noise(x, mean=0, std=0.2)
        return noise_X

    def get_random_mask(self, features, severity):
        mask = torch.rand(features.shape, device=features.device) < severity
        mask = mask.float()
        masked_features = features * (1 - mask)
        return masked_features

    def add_gaussian_noise(self, features, mean=0, std=1): 
        noise = torch.randn_like(features) * std + mean
        noisy_features = features + noise
        return noisy_features





# related masked X recovery
class GCN_recovery(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_view, order):
        super(GCN_recovery, self).__init__()

        self.num_view = num_view
        self.order = order

        self.masked_features = Masked_features(in_dim, hidden_dim, out_dim)

        self.graphencs = GraphEncoder(in_dim, hidden_dim, out_dim, order)

    def forward(self, X, adj_rec, noise_mode):

        for v in range(self.num_view):
            noise_X = self.masked_features(X, noise_mode)
            X_recovery = self.graphencs(noise_X, adj_rec)

        return X_recovery



class MVHGC(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, order, class_num=None, num_view=None):
        super(MVHGC, self).__init__()
        self.num_view = num_view

        self.endecs = nn.ModuleList([   
            EnDecoder(feat_dim, hidden_dim, latent_dim) for _ in range(num_view)
        ])

        self.graphencs = nn.ModuleList([  
            GraphEncoder(feat_dim, hidden_dim, latent_dim, order) for _ in range(num_view)
        ])

        self.graphencs_z = nn.ModuleList([
            GraphEncoder(feat_dim, hidden_dim, latent_dim, order) for _ in range(num_view)
        ])


        self.GCN_recovery = nn.ModuleList([
            GCN_recovery(feat_dim, hidden_dim, latent_dim, num_view, order) for _ in range(num_view)
        ])


        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(num_view)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, latent_dim)))
        for v in range(num_view+1):
            self.register_parameter('centroid_{}'.format(v), self.cluster_layer[v])




    def forward(self, Xs, adjs, weights_h, noise_mode, dataset, ws):


        x_preds = []
        z_norms = []
        A_recs = []
        A_rec_norms = []
        Scores = []
        Scores_dis = []
        hs = []
        qgs = []
        h_from_Ss = []
        h_from_As = []
        str_recs = []

        adj_Ss = []
        adj_Ss_rec = []
        adj_Ss_rec_norm = []
        Xs_recovery = []
        Zs_recovery = []

        for v in range(self.num_view):

            x_pred, z_norm = self.endecs[v](Xs[v])   
            x_preds.append(x_pred)
            z_norms.append(z_norm)

            Score = self.compute_similarity_matrix(z_norm) 
            Scores.append(Score)


            row_means = torch.mean(Score, dim=1, keepdim=True)
            Score_dis = torch.where(Score > row_means, torch.tensor(1, device=Score.device), torch.tensor(0, device=Score.device))  
            Scores_dis.append(Score_dis)


            A_rec = self.construct_adjacency_matrix(Score, threshold=0.5)  
            A_recs.append(A_rec)

            A_rec_norm = self.normalize_adj(A_rec)  
            A_rec_norms.append(A_rec_norm)  




            adj_S = Score + ws[v] * adjs[v]    


            adj_Ss.append(adj_S)                 
            if dataset in ['texas']:
                adj_S_rec_norm = adj_S
                adj_Ss_rec_norm.append(adj_S_rec_norm)

            elif dataset in ['chameleon', 'cornell', 'wisconsin']:
                adj_S_rec_norm = self.normalize_adj(adj_S)  
                adj_Ss_rec_norm.append(adj_S_rec_norm)

            elif dataset in ['acm', 'dblp','imdb']:
                adj_S_map = self.normalize_and_scale(adj_S)  

                adj_S_rec = self.construct_adjacency_matrix(adj_S_map, threshold=0.5)  
                adj_Ss_rec.append(adj_S_rec)

                adj_S_rec_norm = self.normalize_adj(adj_S_rec) 
                adj_Ss_rec_norm.append(adj_S_rec_norm)


            h = self.graphencs[v](z_norm, adj_S_rec_norm)   
            h = F.normalize(h, p=2, dim=-1)
            hs.append(h)


            h_from_S = self.graphencs[v](z_norm, Score)
            h_from_Ss.append(h_from_S)
            h_from_A = self.graphencs[v](z_norm, adjs[v])
            h_from_As.append(h_from_A)


            X_recovery = self.GCN_recovery[v](Xs[v], adj_S, noise_mode)
            X_recovery = F.normalize(X_recovery, p=2, dim=-1)
            Xs_recovery.append(X_recovery)

            Z_recovery = self.GCN_recovery[v](z_norm, adj_S, noise_mode)
            Z_recovery = F.normalize(Z_recovery, p=2, dim=-1)
            Zs_recovery.append(Z_recovery)

            qg = self.predict_distribution(h, v)
            qgs.append(qg)


        h_all = sum(weights_h[v] * hs[v] for v in range(self.num_view)) / sum(weights_h)


        qg = self.predict_distribution(h_all, -1)
        qgs.append(qg)


        return x_preds, z_norms, A_recs, A_rec_norms, Scores, Scores_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, Xs_recovery, Zs_recovery


    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def normalize_adj(self, x):    
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x

    def symmetric_normalize_adjacency(self, adj_matrix):   
        # Calculate degree matrix
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        # Calculate degree matrix's inverse square root
        degree_inv_sqrt = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -0.5))

        # Symmetrically normalize adjacency matrix
        normalized_adj_matrix = torch.mm(torch.mm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)

        return normalized_adj_matrix

    def compute_similarity_matrix(self, X):
        # S = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)   
        # X = F.normalize(X, p=2, dim=1)  
        S = torch.matmul(X, X.t())
        return S

    def construct_adjacency_matrix(self, Score, threshold=0.5):
        # prob_matrix = torch.sigmoid(Scores)
        adjacency_matrix = (Score > threshold).float()
        return adjacency_matrix

    def normalize_matrix(self, matrix):
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def normalize_and_scale(self, matrix, power=2):
        # Data normalization (Min-Max Scaling)
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)

        # Power transformation to expand differences
        scaled_matrix = normalized_matrix ** power

        return scaled_matrix






