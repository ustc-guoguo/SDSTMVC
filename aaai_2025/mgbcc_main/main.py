import argparse
import os.path as path

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary

from granular.base import MVGBList
from granular.granular_loss import MultiviewGCLoss
from model.autoencoder import MultiviewAutoEncoder, Normalize
from model.loss import ContrastiveLoss
from utils.common import load_json, init_torch
from utils.dataset import MultiviewDataset
from utils.score import cluster_metric

from itertools import product
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def run_on_setting(args, **kwargs):
    # 读取全局参数
    global_params = load_json(args.global_config)
    # 设置数据集相关设置的路径，默认与全局配置在同一文件夹下
    if global_params["config"] is None:
        config_path = path.dirname(args.global_config)
        global_params["config"] = path.join(config_path, args.dataset + ".json")
    # 随机数种子
    init_torch(seed=global_params["seed"])
    # 读取数据集特定的参数
    ds_config = load_json(global_params["config"])

    # 优先级：kwargs > ds_config > global_params
    for key in global_params:
        if key not in ds_config:
            ds_config[key] = global_params[key]
    for key in kwargs:
        ds_config[key] = kwargs[key]
    # 读取数据集
    src = ds_config["src"]
    ds_path = path.join(src, ds_config["parent"], args.dataset + ".mat")
    mv_dataset = MultiviewDataset(ds_path, ds_config["device"], views=ds_config["select_views"], normalize=ds_config["normalize"])
    # 构建数据加载器
    batch_size = ds_config["batch_size"]
    if batch_size == -1:
        batch_size = len(mv_dataset)
    # 如果是先基于cpu加载数据，再放到gpu上，则num_workers可以设置大于0
    dataloader = DataLoader(mv_dataset, batch_size, shuffle=True, num_workers=0)
    # 构建模型
    mv_aes = MultiviewAutoEncoder(mv_dataset.view_dims, ds_config["latent_dim"],
                                  ds_config["autoencoder"]["mid_archs"], ds_config["use_linear_projection"])
    # 在编码层后，加一层标准化层
    for v in range(mv_dataset.num_view):
        mv_aes[v].encoder.middle_layers.append(Normalize())
    mv_aes.to(ds_config["device"])
    summary(mv_aes[0], (mv_dataset.view_dims[0],))
    # 优化器
    optimizer = torch.optim.Adam(mv_aes.parameters(),
                                 lr=ds_config["learning_rate"],
                                 weight_decay=ds_config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ds_config["epochs"], eta_min=0.)

    weights = ds_config["loss_weights"]
    kmeans = KMeans(n_clusters=mv_dataset.num_class, n_init="auto", random_state=ds_config["seed"])
    criterion_rec = nn.MSELoss()
    criterion_gra = MultiviewGCLoss()
    criterion_ins = ContrastiveLoss()

    result = {
        "epoch": [],
        "loss_con": [],
        "loss_rec": [],
        "ACC": [],
        "NMI": [],
        "PUR": [],
        "sh": [],
        "ch": [],
        "db": []
    }
    best_feature = None
    best_acc = 0
    best_pred_label = None
    best_gt = None
    for epoch in range(ds_config["epochs"]):
        loss_con_avg = 0
        loss_rec_avg = 0
        mv_aes.train()
        for bid, (x, y) in enumerate(dataloader):
            hs, x_rs = mv_aes(x)
            # 重建损失
            loss_rec = torch.tensor(0., device=ds_config["device"])
            for v in range(mv_dataset.num_view):
                loss_rec += criterion_rec(x[v], x_rs[v])
            if ds_config["p"] > 1:
                # 对本批次的原数据构建粒球
                mv_gblist = MVGBList(hs, y, ds_config["p"])
                # 计算多视图粒球对比损失
                loss_con = criterion_gra(mv_gblist)
            else:
                loss_con, _, _ = criterion_ins(hs)
            loss = loss_con * weights[0] + loss_rec * weights[1]
            loss_con_avg += loss_con.item()
            loss_rec_avg += loss_rec.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_con_avg /= len(dataloader)
        loss_rec_avg /= len(dataloader)
        scheduler.step()
        mv_aes.eval()
        with torch.no_grad():
            hs, _ = mv_aes(mv_dataset.data)
            # 拼接不同视图的特征
            # hs = torch.concat(hs, dim=1).detach().cpu().numpy()
            hs = torch.stack(hs, dim=0).mean(0).detach().cpu().numpy()
            # k_means
            y_pred = kmeans.fit_predict(hs)
            y_true = mv_dataset.labels.cpu().numpy()
            acc, nmi, pur = cluster_metric(y_true, y_pred)
            acc, nmi, pur = round(acc, 4) * 100, round(nmi, 4) * 100, round(pur, 4) * 100
            sh = silhouette_score(hs, y_pred, metric='euclidean')
            ch = calinski_harabasz_score(hs, y_pred)
            db = davies_bouldin_score(hs, y_pred)
            # 记录最佳实验结果
            if acc > best_acc:
                best_acc = acc
                best_feature = hs
                best_pred_label = y_pred
                best_gt = y_true
            print(f"epoch {epoch + 1}, loss_con {round(loss_con_avg, 4):.4f}, loss_rec: {round(loss_rec_avg, 4):.4f}, "
                  f"acc {acc:.2f}%, nmi {nmi:.2f}%, pur {pur:.2f}%, sh {sh:.4f}, ch {ch:.4f}, db {db:.4f}")
            result["epoch"].append(epoch)
            result["loss_con"].append(loss_con_avg)
            result["loss_rec"].append(loss_rec_avg)
            result["ACC"].append(acc)
            result["NMI"].append(nmi)
            result["PUR"].append(pur)
            result["sh"].append(sh)
            result["ch"].append(ch)
            result["db"].append(db)

    return result, best_feature, best_pred_label, best_gt


def run():
    # 读取命令行参数
    parser = argparse.ArgumentParser(description="Command Line Params")
    parser.add_argument("--dataset", type=str, default="Hdigit", help="Dataset used for Training.")
    parser.add_argument("--global_config", type=str, default="./config/granular_config/global.json", help="The path of global config files.")
    args = parser.parse_args()
    # run_on_setting(args)
    # 参数搜索
    latent_dim = [32, 64, 128, 256] # 8, 16, 32, 64,
    p = list(range(1, 17)) # 1, 2,
    learning_rate = [1e-2, 1e-4]
    batch_size = [128, 256, -1]
    normalize = [True, False]
    parameter_iter = product(latent_dim, p, learning_rate, normalize, batch_size)
    dataset = args.dataset
    save_path = "./result/" + dataset + ".dat"
    f = open(save_path, "w")
    for latent_dim, p, learning_rate, normalize, batch_size in parameter_iter:
        result, best_feature, best_pred_label, best_gt = run_on_setting(args, batch_size=batch_size,  latent_dim=latent_dim, p=p, learning_rate=learning_rate, normalize=normalize, dataset=dataset)
        # 需要记录过程中的最佳实验结果和最终实验结果，
        best_epoch = np.argmax(result["ACC"])
        best_result = {
            "epoch": best_epoch,
            "ACC": result["ACC"][best_epoch],
            "NMI": result["NMI"][best_epoch],
            "PUR": result["PUR"][best_epoch],
            "sh": result["sh"][best_epoch],
            "ch": result["ch"][best_epoch],
            "db": result["db"][best_epoch]
        }
        final_result = {
            "epoch": result["epoch"][-1],
            "ACC": result["ACC"][-1],
            "NMI": result["NMI"][-1],
            "PUR": result["PUR"][-1],
            "sh": result["sh"][-1],
            "ch": result["ch"][-1],
            "db": result["db"][-1]
        }
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import random
        colors = ['#f491d2', '#4bc3d9', '#934425', '#fdf04f', '#f9a425', '#4c57bd', '#479ef4', '#ee4f45', '#67ba5c', '#b000ad']
        tsne = TSNE(n_components=2, 
                    perplexity=30, # 每个点考虑多少邻居
                    init='pca',
                    random_state=42)
        feature_2d = tsne.fit_transform(best_feature)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        for i in range(len(set(best_gt))):
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
        import time
        plt.savefig(f'{args.dataset}_tsne_seed_{i+1}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.png') 
        print(f"latent_dim={latent_dim}, p={p}, learning_rate={learning_rate}, normalize={normalize}, batch_size={batch_size}, best_result={best_result}, final_result={final_result}")
        f.flush()
    f.close()



if __name__ == "__main__":
    run()







