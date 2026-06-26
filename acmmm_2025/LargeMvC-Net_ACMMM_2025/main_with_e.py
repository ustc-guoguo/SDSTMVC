from __future__ import print_function, division
import random
from tqdm import tqdm

from model_with_e import DBONet_with_E
from util.clusteringPerformance import StatisticClustering
from util.utils import  features_to_adj
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from util.utils import normalization, standardization, normalize
import os
import scipy.io as sio
import time

def train(data, features, epoch, block,  labels, n_view, n_clusters, model, optimizer, scheduler, device):
    acc_max = 0.0
    res = []
    for i in range(n_view):
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
    criterion = nn.MSELoss()
    best_feature = None
    best_pred_label = None
    with tqdm(total=epoch, desc="Training") as pbar:
        start_time = time.time()
        for i in range(epoch):
            model.train()
            optimizer.zero_grad()
            output_z,  output_d, output_e = model(features)
            loss_rec = torch.Tensor(np.array([0])).to(device)
            for k in range(n_view):
                    exec("loss_rec+=criterion(output_d[{}].mm(output_z), features[{}])".format(k, k))  # loss function
            loss =  loss_rec
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            Rec_loss = loss_rec.cpu().detach().numpy()
            output_zz = output_z.detach().cpu().numpy().T
            # get clustering result
            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_zz, labels, n_clusters)
            if (ACC[0] > acc_max):
                acc_max = ACC[0]
                best_feature = output_zz
                best_pred_label = labels
                res = []
                for item in [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]:
                    res.append("{}({})".format(item[0] * 100, item[1] * 100))
            pbar.update(1)
            print({"Rec_loss": "{:.6f}".format(Rec_loss[0]),
                   'ACC': '{:.2f} | {:.2f}'.format(ACC[0] * 100, acc_max * 100)})
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time: {total_time:.6f} S")
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import random
        colors = ['#f491d2', '#4bc3d9', '#934425', '#fdf04f', '#f9a425', '#4c57bd', '#479ef4', '#ee4f45', '#67ba5c', '#b000ad']
        tsne = TSNE(n_components=2, 
                    perplexity=30, # 每个点考虑多少邻居
                    init='pca',
                    random_state=i+1)
        feature_2d = tsne.fit_transform(best_feature)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        for i in range(n_clusters):
            cluster_data = feature_2d[best_pred_label == i]
            
            col = colors[i] if i < len(colors) else '#'+random.choice('0123456789ABCDEF')*6
            
            ax.scatter(cluster_data[:, 0], 
                        cluster_data[:, 1], 
                        label=f'Cluster {i + 1}', 
                        s=10,# 点大小
                        # alpha=0.8,
                        color=col)
        # 去坐标
        ax.set_xticks([])
        ax.set_yticks([])
        # 去边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{data}_tsne.png') 
    return res



def main(data, args):
    # Clustering evaluation metrics
    SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']

    seed = args.seed
    lr = args.lr
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Loading dataset and initializing
    X, Z_init, D_init, E_init, m, n_view, num_sample,labels,n_clusters = features_to_adj(data, args.path + args.data_path)

    print("anchor size:{}, view size:{}, sample size:{}".format(m, n_view, num_sample))

    # load model
    model = DBONet_with_E(m, n_view, args.block, args.thre1, args.thre2, Z_init, D_init, E_init, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, 
                                                           min_lr=1e-8)
    # Training
    res = train(data, X, args.epoch, args.block,  labels, n_view, n_clusters, model, optimizer, scheduler, device)

    print("{}:{}\n".format(data, dict(zip(SCORE, res))))
    with open(args.save_file,'a') as f:
        f.write("{}:{}\n".format(data, dict(zip(SCORE, res))))  # save result