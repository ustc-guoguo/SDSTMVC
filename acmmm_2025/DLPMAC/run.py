import argparse
import time
import random
import config
from utils import *
import math
import torch,gc
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader
from models import *
from losstrace import *
from Clustering import *
from tqdm import tqdm
parser = argparse.ArgumentParser(description='M2M in PyTorch')
parser.add_argument('--data', default=12, type=int,
                    help='choice of dataset, 0-HW,1-3Sources,2BBC,3-Scene15, 4-Caltech101,5-ORL_mtv,6-Caltech_7,7-Reuters_dim10,'
                         '8-20newsgroups,9-100leaves,10-BBC4,11-MSRCv1,12-BDGP,13-HandWritten,14-yale_mtv,15-Wikipedia-test,16-Movies,17-Prokaryotic,18-ALOI,19-flower17,20-NUSWIDE')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('--prebs', type=int, default=1, help='num of pretrain batch')
parser.add_argument('--Cbs', type=int, default=2, help='num of Cbs batch')
parser.add_argument('-e', '--epochs', default='100', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='0.0001', type=float, help='learning rate of adam')
parser.add_argument('-ap', '--aligned-prop', default='1.0', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx to use.')
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float,
                    help='originally complete proportions in the partially sample-missing data')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('-s', '--start-fine', default=True, type=bool, help='flag to start use robust loss or not')
parser.add_argument('-np', '--neg-num', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels')
parser.add_argument('-r', '--robust', default=1, type=int, help='use our robust loss or not')
parser.add_argument('--lamda', type=int, default=1, help='Hyperparameters')

dim=0
# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []


class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, use_robust_loss, args):
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        if use_robust_loss == 1:
            if args.start_fine:
                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, 0.5) * (0.5*margin - pair_dist), min=0.0), 2)
            else:
                loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss

def train(train_loader, model, criterion1,criterion2, optimizer, epoch, args):

    model.train()
    time0 = time.time()
    loss_value = 0
    for batch_idx, (x0, x1, labels, real_labels) in enumerate(train_loader):
        x0, x1,  labels, real_labels = x0.to(args.gpu), x1.to(args.gpu), labels.to(
            args.gpu), real_labels.to(args.gpu)

        h0, h1, d0, d1 = model(x0.view(x0.size()[0], -1), x1.view(x1.size()[0], -1))

        #
        x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
        # x01,x11=x0.view(x0.size()[0], -1).clone(),x1.view(x1.size()[0], -1).clone()

        # loss = criterion1(x0, d0)
        # loss += criterion1(x1, d1)
        pair_dist = F.pairwise_distance(h0, h1)  # use Euclidean distance to measure similarity

        loss = criterion2(pair_dist, labels, args.margin, args.robust, args)
        loss_value += loss.item()
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0



    return epoch_time

def align(xs,model,optimizer,args):
    model.eval()
    for v in range(2):
        xs[v]=torch.tensor(xs[v]).T
    for v in range(2):
        xs[v] = xs[v].to(args.gpu)
    print(np.shape(xs[0]))
    
    optimizer.zero_grad()
    h0, h1, d0, d1 = model(xs[0], xs[1])
    sim_matrix = cosineSimilarty(h0.to('cpu').detach(), h1.to('cpu').detach())
    max_values, max_indices = torch.max(sim_matrix, dim=1)

    # 输出结果
    print("Max values per row_fulltrain:", max_values)  # 每行的最大值
    print("Indices of max values per row_fulltrain:", max_indices)  # 每行最大值对应的索引
    print(f"\nsim_matrix: {sim_matrix}")
def main():
    j = 0
    while j < 1:
        j = j + 1
        args = parser.parse_args()
        data_name = ['HandWritten', '3Sources', 'BBCsports', 'Scene15', 'Caltech101', 'ORL_mtv', 'Caltech101_7', 'Reuters_dim10',
                 '20NewsGroups','100leaves','BBC4','MSRCv1','BDGP','HandWritten','yale_mtv','Wikipedia-test','Movies','Prokaryotic','ALOI','flower17','NUSWIDE']
        NetSeed = random.randint(1,1000)
        # NetSeed=422
        print(NetSeed)
        # NetSeed = random.seed()
        np.random.seed(NetSeed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(NetSeed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(NetSeed)  # 为当前GPU设置随机种子
        print(data_name[args.data])
        train_pair_loader, all_data, all_label, all_label_X, all_label_Y, pretrain_pair_loader, dim, class_num, divide_seed = (
            loader(args.batch_size, args.aligned_prop,args.complete_prop,args.neg_num,args.noisy_training,data_name[args.data]))
        config.classes=class_num
        model = SdA(config).to(args.gpu)
        criterion1 = nn.MSELoss().to(args.gpu)
        criterion2 = NoiseRobustLoss().to(args.gpu)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

        CAR_list = []
        acc_list, nmi_list, ari_list, f_list, f1_list, pre_list, pre2_list, rec_list, pur_list = [], [], [], [], [], [], [], [], []
        train_time = 0
        all_data[0], all_data[1] = torch.from_numpy(all_data[0]), torch.from_numpy(all_data[1])
        #
        pretrain(model, optimizer, pretrain_pair_loader, criterion1, args)
        # align(all_data,model,optimizer,args)
        # train
        best_data = None
        best_pred_label = None
        best_accuracy = 0   
        for i in range(0, args.epochs + 1):
            if i == 0:
                with torch.no_grad():
                    epoch_time = train(train_pair_loader,model, criterion1, criterion2, optimizer, i, args)
            else:
                epoch_time = train(train_pair_loader,model, criterion1,criterion2 ,optimizer, i,args)
            v0, v1, pred_label, alignment_rate = tiny_infer(model, args.gpu, all_data, all_label_X, all_label_Y)
            print(alignment_rate)

            CAR_list.append(alignment_rate)
            # p0,p1=v0.copy(),v1.copy()
            # p0,p1=torch.tensor(p0),torch.tensor(p1)
            # Pseudo0, Pseudo1 = F.softmax(p0, dim=1),F.softmax(p1,dim=1)
            # Pseudo0, Pseudo1=np.array(Pseudo0),np.array(Pseudo1)
            # v0 = np.concatenate((v0,Pseudo0), axis=1)
            # v1 = np.concatenate((v1, Pseudo1), axis=1)

            data = []
            data.append(v0)
            data.append(v1)

            y_pred, ret, accuracy, nmi, ari, f_score, f_score2, precision, precision2, recall, purity = Clustering(data,
                                                                                                                   pred_label)
            if i % 2 == 0:
                # tsne_visualize(data, 'BDGP', 'root_location', pred_label)
        
                print(accuracy, nmi, ari, f_score, f_score2, precision, precision2, recall, purity)
                # logging.info("******** testing ********")
                # logging.info(
                #     "CAR={} kmeans: acc={} nmi={} ari={}".format(round(alignment_rate, 4), ret['kmeans']['accuracy'],
                #                                                  ret['kmeans']['NMI'], ret['kmeans']['ARI']))
            acc_list.append(ret['kmeans']['ACC'])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_data = data
                best_pred_label = pred_label
            nmi_list.append(ret['kmeans']['NMI'])
            ari_list.append(ret['kmeans']['ARI'])
            f_list.append(ret['kmeans']['F1'])
            f1_list.append(ret['kmeans']['F2'])
            pre_list.append(ret['kmeans']['PRE'])
            pre2_list.append(ret['kmeans']['PRE2'])
            rec_list.append(ret['kmeans']['REC'])
            pur_list.append(ret['kmeans']['PUR'])
        print('alignrate:',max(CAR_list))
        print('ACC:', max(acc_list))
        print("NMI:", max(nmi_list))
        print("ARI:", max(ari_list))
        print("F1:", max(f_list))
        print("F2:", max(f1_list))
        print("PRE:", max(pre_list))
        print("PRE2:", max(pre2_list))
        print("REC:", max(rec_list))
        print("PUR:", max(pur_list))
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        colors = ['#f491d2', '#4bc3d9', '#934425', '#fdf04f', '#f9a425', '#4c57bd', '#479ef4', '#ee4f45', '#67ba5c', '#b000ad']
        tsne = TSNE(n_components=2, 
                    perplexity=30, # 每个点考虑多少邻居
                    init='pca',
                    random_state=i+1)
        best_data = np.concatenate(best_data[:], axis=1)
        feature_2d = tsne.fit_transform(best_data)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        for i in range(class_num):
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
        plt.savefig(f'{data_name[args.data]}_tsne_seed_{i+1}.png') 
        # logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))
        #


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    main()
