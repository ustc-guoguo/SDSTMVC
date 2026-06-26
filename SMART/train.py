import swanlab
swanlab.login(api_key="K1HxjHEB393lXCLOYPrAe", save=True)
import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import evaluate
from clustering import clustering
from utils import compare_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


def train(model, optimizer, data_loader, config, logger, seed, device):
    max_acc = 0.0
    max_acc_pred_label = None
    max_acc_feature_fu = None
    clusterImages = []
    loss_metrics = {'loss': [], 'loss_rec': [], 'loss_cvc': [], 'loss_ncl': [],
                    'acc': [], 'nmi': [], 'ari': [], 'pur': []}

    is_best = False
    best_fu = (0.0, 0.0, 0.0, 0.0, is_best)

    # training
    for epoch in tqdm(range(config['epochs'])):
        loss_, loss_rec_, loss_cvc_, loss_ncl_ = 0., 0., 0., 0.
        feat_fu, feat1, feat2, label1_list, label2_list, flag_list = [], [], [], [], [], []
        for i, (x1, x2, label1, label2, flag_batch, indices_batch) in enumerate(data_loader): # torch.Size([2500, 79]) torch.Size([2500, 1750])
            x1, x2, flag_batch = x1.to(device), x2.to(device), flag_batch.to(device)
            # Forward
            model.train()
            emb1, emb2, x1_hat, x2_hat = model(x1, x2) # torch.Size([2500, 30]) torch.Size([2500, 30]) torch.Size([2500, 79]) torch.Size([2500, 1750])

            # ==============================================================================

            # 1. Within-view Reconstruction
            loss_rec = model.reconstruction_loss(x1, x2, x1_hat, x2_hat)

            # 2. View Distribution Alignment
            score_all = model.cross_corr_coef_matrix(emb1, emb2) # torch.Size([2500, 2500])
            score_aligned = score_all[flag_batch, :]
            score_aligned = score_aligned[: , flag_batch] # torch.Size([1250, 1250])
            loss_cvc = model.VDA_loss(emb1[flag_batch],
                                      emb2[flag_batch],
                                      corr_coef_matrix=score_aligned
                                                  )

            # 3. Semantic Matching Contrastive Learning
            score_aligned_diag = torch.diag(score_aligned.detach().clone()) # torch.Size([1250])
            # adj with no gradient here
            adj = model.get_semantic_graph(
                score_all.detach().clone(),
                flag_batch,
                score_aligned_diag,
                topk=config.get('semantic_topk', None), # 新增
                mutual=config.get('semantic_mutual', False), # 新增
            )
            H1, H2 = model.high_level_project(emb1, emb2)
            smc_tau = config.get('smc_tau', config['tau'])
            loss_ncl = model.SMC_loss(H1, H2, adj, tau=smc_tau)

            # ==============================================================================
            loss = loss_rec + loss_ncl * config['lambda2'] + loss_cvc * config['lambda1']
            loss_rec_ += loss_rec.item()
            loss_cvc_ += loss_cvc.item()
            loss_ncl_ += loss_ncl.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()
            swanlab.log({
                f"loss_{seed}/loss_all": loss.item(),
                f"loss_{seed}/loss_rec": loss_rec.item(),
                f"loss_{seed}/loss_cvc": loss_cvc.item(),
                f"loss_{seed}/loss_ncl": loss_ncl.item(),
            })
            H2_hat = torch.matmul(adj, H2.detach())
            H2_hat = F.normalize(H2_hat, p=2, dim=-1)
            feat_fu.append(torch.concat((H1.detach(), H2_hat), dim=-1))
            label1_list.append(label1)
            label2_list.append(label2)
            flag_list.append(flag_batch.cpu())

        if torch.isnan(torch.tensor(loss_)):
            break
        loss_metrics['loss'].append(loss_)

        feature_fu = torch.cat(feat_fu, dim=0)
        label_true1 = torch.cat(label1_list).numpy()
        label_true2 = torch.cat(label2_list).numpy()

        # 聚类
        label_pred, _, _ = clustering(feature=feature_fu,
                                      cluster_num=config['n_classes'],
                                      device=device)
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
    # 聚类可视化——最佳聚类结果
    if max_acc_pred_label is not None:
        colors = ['#f491d2', '#4bc3d9', '#934425', '#fdf04f', '#f9a425', '#4c57bd', '#479ef4', '#ee4f45', '#67ba5c', '#b000ad']
        tsne = TSNE(n_components=2, 
                    perplexity=30, # 每个点考虑多少邻居
                    init='pca',
                    random_state=seed)
        feature_2d = tsne.fit_transform(max_acc_feature_fu.detach().cpu().numpy())
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        for i in range(config['n_classes']):
            cluster_data = feature_2d[max_acc_pred_label == i]
            
            col = colors[i] if i < len(colors) else '#'+random.choice('0123456789ABCDEF')*6
            
            ax.scatter(cluster_data[:, 0], 
                        cluster_data[:, 1], 
                        label=f'Cluster {i + 1}', 
                        s=10,# 点大小
                        # alpha=0.8,
                        color=col)
        # 显示图例
        # plt.legend()

        # 去坐标
        ax.set_xticks([])
        ax.set_yticks([])
        # 去边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.tight_layout()
        clusterImages.append(swanlab.Image(plt, caption=f'best_epoch-{best_epoch}'))

    swanlab.log({f"Visiual_{seed}/cluster": clusterImages})
    logger.write(f'{best_fu[0]*100:.2f}|{best_fu[1]*100:.2f}|{best_fu[2]:.4f}|{best_fu[3]:.4f}')
    res = []
    res.append(swanlab.Text("ACC", caption=f"{best_fu[0]:.4f}"))
    res.append(swanlab.Text("NMI", caption=f"{best_fu[1]:.4f}"))
    res.append(swanlab.Text("ARI", caption=f"{best_fu[2]:.4f}"))
    res.append(swanlab.Text("PUR", caption=f"{best_fu[3]:.4f}"))
    res.append(swanlab.Text("RES", caption=f"ACC:{best_fu[0]*100:.2f}, NMI:{best_fu[1]*100:.2f}, ARI:{best_fu[2]*100:.2f}, PUR:{best_fu[3]*100:.2f}"))
    swanlab.log({f"Metrics_{seed}/Result": res})
    return best_fu
