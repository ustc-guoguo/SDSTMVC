import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import evaluate
from clustering import clustering
from utils import compare_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
import random
import swanlab
from PIL import Image


swanlab.login(api_key="K1HxjHEB393lXCLOYPrAe", save=True)

def train(model, optimizer, data_loader, config, logger, seed, device):
    max_acc = 0.0
    max_acc_pred_label = None
    max_acc_feature_fu = None
    loss_metrics = {'loss': [], 'loss_rec': [], 'loss_cvc': [], 'loss_ncl': [],
                    'acc': [], 'nmi': [], 'ari': [], 'pur': []}

    is_best = False
    best_fu = (0.0, 0.0, 0.0, 0.0, is_best)

    distributionImages = []
    clusterImages = []
    # training
    for epoch in tqdm(range(config['epochs'])): 
        if (epoch+1) % 100 == 0 and epoch > 0:
            t_sne_distribution_x1 = torch.empty((0, config['emb_dim']), device=device, requires_grad=False)
            t_sne_distribution_x2 = torch.empty((0, config['emb_dim']), device=device, requires_grad=False)

        loss_, loss_rec_, loss_cvc_, loss_ncl_ = 0., 0., 0., 0.
        feat_fu, feat1, feat2, label1_list, label2_list, flag_list = [], [], [], [], [], []


        for i, (x1, x2, label1, label2, flag_batch, indices_batch) in enumerate(data_loader):
            
            x1, x2, flag_batch = x1.to(device), x2.to(device), flag_batch.to(device)

            ## Forward
            model.train()
            emb1, emb2, x1_hat, x2_hat = model(x1, x2) # torch.Size([2500, 30])
            if (epoch+1) % 100 == 0 and epoch > 0:
                t_sne_distribution_x1 = torch.cat((t_sne_distribution_x1, emb1), dim=0)
                t_sne_distribution_x2 = torch.cat((t_sne_distribution_x2, emb2), dim=0)

            # 1. Within-view Reconstruction
            loss_rec = model.reconstruction_loss(x1, x2, x1_hat, x2_hat)


            # KL divergence
            loss_kl = F.kl_div(
                F.log_softmax(emb1, dim=-1),
                F.softmax(emb2, dim=-1),
                reduction='batchmean'
            ) + F.kl_div(
                F.log_softmax(emb2, dim=-1),
                F.softmax(emb1, dim=-1),
                reduction='batchmean'
            )


            # JS divergence
            m = 0.5 * (F.softmax(emb1, dim=-1) + F.softmax(emb2, dim=-1)) # 计算平均分布 M
            kl_p_m = F.kl_div(F.log_softmax(emb1, dim=-1), m, reduction='batchmean')
            kl_q_m = F.kl_div(F.log_softmax(emb2, dim=-1), m, reduction='batchmean')
            loss_js = 0.5 * (kl_p_m + kl_q_m)



            # 2. View Distribution Alignment
            score_all = model.cross_corr_coef_matrix(emb1, emb2)
            score_aligned = score_all[flag_batch, :]
            score_aligned = score_aligned[: , flag_batch]
            loss_cvc = model.VDA_loss(emb1[flag_batch],
                                      emb2[flag_batch],
                                      corr_coef_matrix=score_aligned)

            # 3. Semantic Matching Contrastive Learning
            score_aligned_diag = torch.diag(score_aligned.detach().clone())
            adj = model.get_semantic_graph(score_all.detach().clone(), # adj with no gradient here
                                         flag_batch,
                                         score_aligned_diag)
            H1, H2 = model.high_level_project(emb1, emb2)
            loss_ncl = model.SMC_loss(H1, H2, adj, tau=config['tau'])


            ## total loss
            # loss_rec
            # loss_cvc
            # loss_ncl
            # loss_kl
            # loss_js
            loss = loss_rec + loss_cvc * config['lambda1'] + 0.2 * loss_js #  + loss_ncl * config['lambda2']

            swanlab.log({
                f"loss_{seed}/loss_all": loss.item(),
                f"loss_{seed}/loss_rec": loss_rec.item(),
                f"loss_{seed}/loss_cvc": loss_cvc.item(),
                f"loss_{seed}/loss_ncl": loss_ncl.item(),
                f"loss_{seed}/loss_kl": loss_kl,
                f"loss_{seed}/loss_js": loss_js,
            })
            loss_rec_ += loss_rec.item()
            loss_cvc_ += loss_cvc.item()
            loss_ncl_ += loss_ncl.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ += loss.item()


            ## clustering
            H2_hat = torch.matmul(adj, H2.detach())
            H2_hat = F.normalize(H2_hat, p=2, dim=-1)
            feat_fu.append(torch.concat((H1.detach(), H2_hat), dim=-1))
            label1_list.append(label1)
            label2_list.append(label2)
            flag_list.append(flag_batch.cpu())
        


        ################################ t-SNE visualization
        if (epoch+1) % 100 == 0 and epoch > 0:
            try:
                tsne = TSNE(n_components=2, random_state=seed)
                t_sne_distribution_x1 = tsne.fit_transform(t_sne_distribution_x1.detach().cpu().numpy())
                t_sne_distribution_x2 = tsne.fit_transform(t_sne_distribution_x2.detach().cpu().numpy())
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
                ax.scatter(t_sne_distribution_x1[:, 0], t_sne_distribution_x1[:, 1], c='blue', cmap='tab10', label='View 1')
                ax.scatter(t_sne_distribution_x2[:, 0], t_sne_distribution_x2[:, 1], c='red', cmap='tab10', label='View 2')
                ax.legend()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                distributionImages.append(swanlab.Image(fig, caption=f'epoch-{epoch}'))
            except Exception as e:
                pass
        ##############################



        if torch.isnan(torch.tensor(loss_)):
            break
        loss_metrics['loss'].append(loss_)

        feature_fu = torch.cat(feat_fu, dim=0)
        label_true1 = torch.cat(label1_list).numpy()
        label_true2 = torch.cat(label2_list).numpy()

        label_pred, _, _ = clustering(feature=feature_fu,
                                      cluster_num=config['n_classes'],
                                      device=device)
        
        

        ################################ 聚类可视化
        if (epoch+1) % 100 == 0 and epoch > 0:
            try:
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan', 'magenta', 'yellow']
                tsne = TSNE(n_components=2, random_state=seed)
                feature_2d = tsne.fit_transform(feature_fu.detach().cpu().numpy())
                ax = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
                for i in range(config['n_classes']):
                    cluster_data = feature_2d[label_pred == i]
                    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=30, color=colors[i])
                plt.legend()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                clusterImages.append(swanlab.Image(plt, caption=f'epoch-{epoch}'))
            except Exception as e:
                pass
        ################################
        


        acc, nmi, ari, pur = evaluate(label_true1, label_pred)
        if acc > max_acc:
            max_acc = acc
            max_acc_pred_label = label_pred
            max_acc_feature_fu = feature_fu

        loss_metrics['acc'].append(acc)
        loss_metrics['nmi'].append(nmi)
        loss_metrics['ari'].append(ari)
        loss_metrics['pur'].append(pur)
        swanlab.log({
                f"Metrics_{seed}/acc": acc,
                f"Metrics_{seed}/nmi": nmi,
                f"Metrics_{seed}/ari": ari,
                f"Metrics_{seed}/pur": pur,
        })

        is_best = False
        score = (acc, nmi, ari, pur, is_best)
        best_fu = compare_score(score, best_fu)
        if best_fu[-1]:
            best_epoch = epoch+1

    ################################ 聚类可视化——最佳聚类结果
    if max_acc_pred_label is not None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan', 'magenta', 'yellow']
        tsne = TSNE(n_components=2, random_state=seed)
        feature_2d = tsne.fit_transform(max_acc_feature_fu.detach().cpu().numpy())
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        for i in range(config['n_classes']):
            cluster_data = feature_2d[max_acc_pred_label == i]
            col = colors[i] if i < len(colors) else '#'+random.choice('0123456789ABCDEF')*6
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=30, color=col)
        plt.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        clusterImages.append(swanlab.Image(plt, caption=f'best_epoch-{best_epoch}'))
    ################################

    swanlab.log({f"Visiual_{seed}/distribution": distributionImages})
    swanlab.log({f"Visiual_{seed}/cluster": clusterImages})
    logger.write(f'  ACC {best_fu[0]:.4f}, NMI {best_fu[1]:.4f}, ARI {best_fu[2]:.4f}, PUR {best_fu[3]:.4f}')
    res = []
    res.append(swanlab.Text("ACC", caption=f"{best_fu[0]:.4f}"))
    res.append(swanlab.Text("NMI", caption=f"{best_fu[1]:.4f}"))
    res.append(swanlab.Text("ARI", caption=f"{best_fu[2]:.4f}"))
    res.append(swanlab.Text("PUR", caption=f"{best_fu[3]:.4f}"))
    swanlab.log({f"Metrics_{seed}/Result": res})
    return best_fu
