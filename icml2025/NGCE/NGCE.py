import argparse
import os.path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans 
from sklearn.neighbors import NearestNeighbors
from utils import load_data, normalize_weight, cal_homo_ratio
from models import EnDecoder, MVHGC, GNN
from evaluation import eva
from settings import get_settings
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def knn(features, queries, k=20):
    dists = torch.cdist(queries, features)  
    _, indices = torch.topk(dists, k, largest=False, sorted=True)
    nearest_features = features[indices]  
    
    return nearest_features

def batch_contrastive_loss(f_x, f_x_plus, f_x_neg):
    if f_x.dim() == 1:
        f_x = f_x.unsqueeze(0)  
    f_x = f_x.unsqueeze(1)  
    f_x_plus_transposed = f_x_plus.transpose(1, 2)  
    pos_similarities = torch.bmm(f_x, f_x_plus_transposed).squeeze(1)  
    f_x_transposed = f_x.transpose(1, 2)  
    neg_similarities = torch.bmm(f_x_neg, f_x_transposed).squeeze(1)  

    pos_exp = torch.exp(pos_similarities)  
    neg_exp = torch.exp(neg_similarities)  
    if pos_exp.dim() == 1:
        pos_exp = pos_exp.unsqueeze(1)  
    if neg_exp.dim() == 1:
        neg_exp = neg_exp.unsqueeze(1)  
    pos_exp_sum = torch.sum(pos_exp, dim=1, keepdim=True)  
    neg_exp_sum = torch.sum(neg_exp, dim=1, keepdim=True)  
    denominator = pos_exp_sum + neg_exp_sum  
    log_probs = torch.log(pos_exp_sum / denominator)  
    loss = -torch.mean(log_probs)
    return loss

_GLOBAL_CLASS_NUM_HOLDER = [None] 

def run_kmeans(data, n_clusters=None, n_init=5): 
    global kmeans 
    if 'kmeans' not in globals() or not callable(globals()['kmeans']):
        from kmeans_pytorch import kmeans as km_pytorch_kmeans
        kmeans = km_pytorch_kmeans 

    if n_clusters is None:
        if _GLOBAL_CLASS_NUM_HOLDER[0] is not None:
            n_clusters = _GLOBAL_CLASS_NUM_HOLDER[0]
        else:
            raise ValueError("n_clusters must be specified or class_num must be globally set for run_kmeans")

    cluster_ids, centers = kmeans(
        X=data,
        num_clusters=n_clusters,
        distance='euclidean',
        device=data.device,
    )
    return cluster_ids.cpu().numpy(), centers

for matadata in ['dblp']:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=matadata, help='datasets: acm, dblp, texas, chameleon, wisconsin, cornell, imdb')
    parser.add_argument('--train', type=bool, default=True, help='training mode')
    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    args = parser.parse_args()

    dataset = args.dataset
    train = args.train
    cuda_device = args.cuda_device
    use_cuda = args.use_cuda

    settings = get_settings(dataset)

    path = settings.path
    order = settings.order
    weight_soft = settings.weight_soft
    T0 = settings.T0
    noise_mode = settings.noise_mode
    w = settings.w

    nlayers = settings.nlayers
    hid_dim = settings.hid_dim

    hidden_dim = settings.hidden_dim
    latent_dim = settings.latent_dim

    epoch = settings.epoch
    patience = settings.patience

    lr = settings.lr
    weight_decay = settings.weight_decay
    update_interval = settings.update_interval
    random_seed = settings.random_seed
    torch.manual_seed(random_seed)

    labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path)

    for v in range(graph_num):
        r = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
        print(r)
    print('dataset: {}'.format(dataset))
    print('nodes: {}'.format(shared_feature_label.shape[0]))
    print('features: {}'.format(shared_feature_label.shape[1]))
    print('class: {}'.format(labels.max() + 1))
    print('order: {}'.format(order))
    print('w: {}'.format(w))

    feat_dim = shared_feature.shape[1]
    class_num = labels.max().item() + 1
    _GLOBAL_CLASS_NUM_HOLDER[0] = class_num 
    y = labels.cpu().numpy()

    xs = []
    for v in range(graph_num):
        xs.append(shared_feature_label)

    model = MVHGC(feat_dim, hidden_dim, latent_dim, order, class_num=class_num, num_view=graph_num)

    if use_cuda:
        torch.cuda.set_device(cuda_device)
        torch.cuda.manual_seed(random_seed)
        model = model.cuda()
        adjs_labels = [a.cuda() for a in adjs_labels]
        xs = [x.cuda() for x in xs]
        shared_feature = shared_feature.cuda()
        shared_feature_label = shared_feature_label.cuda()
    device = adjs_labels[0].device

    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if train:
        # =========================================Train=============================================================
        print('Begin trains...')

        weights_h = []
        weighh = [1e-12 for i in range(graph_num)]
        weights_h = normalize_weight(weighh)

        with torch.no_grad():
            model.eval()
            ws = []
            for v in range(graph_num):
                ws.append(w)

            x_preds, z_norms, A_recs, A_rec_norms, Scores, Scores_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, Xs_recovery, Zs_recovery = model(xs, adjs_labels, weights_h, noise_mode, dataset, ws)
            from kmeans_pytorch import kmeans 

            for v in range(graph_num):
                y_pred, centers = run_kmeans(hs[v], n_clusters=class_num)
                model.cluster_layer[v].data = centers.to(device)
            
            y_pred, centers = run_kmeans(hs[-1], n_clusters=class_num)
            model.cluster_layer[-1].data = centers.to(device)
            
            for v in range(graph_num):
                y_eval_all, _ = run_kmeans(h_all, n_clusters=class_num)
                
                r_S = cal_homo_ratio(
                    Scores[v].detach().cpu().numpy(), 
                    y_eval_all, 
                    self_loop=True
                )
                r_A = cal_homo_ratio(
                    adjs_labels[v].detach().cpu().numpy(), 
                    y_eval_all, 
                    self_loop=True
                )
                ws[v] = r_A  
            # kmeans = KMeans(n_clusters=class_num, n_init=5)
            # for v in range(graph_num):
            #       y_pred = kmeans.fit_predict(hs[v].data.cpu().numpy())
            #       model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
            # y_pred = kmeans.fit_predict(hs[-1].data.cpu().numpy())
            # model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_).to(device)
            
            # for v in range(graph_num):
            #       kmeans_S = KMeans(n_clusters=class_num, n_init=5)
            #       y_eval_all = kmeans_S.fit_predict(h_all.detach().cpu().numpy())
            #       r_S = cal_homo_ratio(Scores[v].detach().cpu().numpy(), y_eval_all, self_loop=True)
            #       r_A = cal_homo_ratio(adjs_labels[v].detach().cpu().numpy(), y_eval_all, self_loop=True)
            #       w = r_A
            #       ws[v] = w

        bad_count = 0
        best_sum_metrics_val = 0.0  
        acc_at_best_sum = 0.0
        nmi_at_best_sum = 0.0
        ari_at_best_sum = 0.0
        f1_at_best_sum = 0.0
        epoch_at_best_sum = 0

        max_acc_overall = 0.0
        max_nmi_overall = 0.0
        max_ari_overall = 0.0
        max_f1_overall = 0.0
        HRs = []
        losses = []
        for epoch_num in range(epoch):
            model.train()

            loss_kl = 0.
            loss_re_x =0.
            loss_kl_g = 0.
            kl_step = 1.
            kl_max = 10000
            loss = 0.
            l = 0.0
            loss_Xs_recovery = 0.0
            loss_Zs_recovery = 0.0
            loss_hs_recovery = 0.0
            loss_contrastive = 0.0 

            x_preds, z_norms, A_recs, A_rec_norms, Scores, Scores_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, Xs_recovery, Zs_recovery = model(xs, adjs_labels, weights_h, noise_mode, dataset, ws)
            
            from kmeans_pytorch import kmeans 

            with torch.no_grad():
                cluster_ids, _ = kmeans(
                    X=h_all,                      
                    num_clusters=class_num,
                    distance='euclidean',
                    device=h_all.device,        
                )
                y_prim = cluster_ids.cpu().numpy()
                pseudo_label = y_prim

                for v in range(graph_num):
                    cluster_ids_v, _ = kmeans( 
                        X=hs[v],
                        num_clusters=class_num,
                        distance='euclidean',
                        device=hs[v].device,
                    )
                    y_pred_v = cluster_ids_v.cpu().numpy() 
                    
                    a = eva(y_prim, y_pred_v, visible=False, metrics='acc')
                    weighh[v] = a              

                # for v in range(graph_num):
                #     data = hs[v].detach().cpu().numpy()  
                #     kmeans = KMeans(n_clusters=k, random_state=0)
                #     y_pred = kmeans.fit_predict(data)  
                #     K1 = 30 
                #     K2 = 10
                #     nn = NearestNeighbors(n_neighbors=K1+1, algorithm='auto').fit(data)
                #     distances, indices = nn.kneighbors(data) 
                #     indices = indices[:, 1:] 
                #     negative_samples_indices = [[] for _ in range(len(data))]  

                #     for i in range(len(data)):
                #         negative_indices = np.where(y_pred != y_pred[i])[0] 
                #         negative_samples_indices[i] = negative_indices
                #     if len(negative_indices) > K2:
                #         chosen_negative_indices = np.random.choice(negative_indices, K2, replace=False)  
                #     else:
                #         chosen_negative_indices = negative_indices

                # if epoch_num > T0 and v==0:
                #     contrastive_total_loss = 0.0
                # if epoch_num > T0:
                #     # triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
                #     # contrastive_total_loss = 0.0
                #     # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
                #     for i in range(class_num):
                #         class_mask = (y_prim == i)
                #         class_samples = h_all[class_mask]  

                #         positive = cluster_centers[i] 

                #         neg_mask = torch.ones(class_num, dtype=torch.bool)
                #         neg_mask[i] = False
                #         negative_centers = cluster_centers[neg_mask] 

                #         for sample in class_samples:
                #             # for negative in negative_centers:
                #                 # loss1 = triplet_loss(sample.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                #                 # contrastive_total_loss += loss1.item()
                #             contrastive_total_loss += batch_contrastive_loss(sample.unsqueeze(0), positive.unsqueeze(0).unsqueeze(0), negative_centers.unsqueeze(0))
        
                #     loss_contrastive = contrastive_total_loss / (len(h_all) * (class_num - 1))
            weights_h = normalize_weight(weighh, p=weight_soft)
            with torch.no_grad():
                cluster_ids_eval, cluster_centers_eval = kmeans( # Renamed to avoid conflict
                    X=h_all,                
                    num_clusters=class_num,
                    distance='euclidean',        
                    device=h_all.device,        
                )
                y_eval_all = cluster_ids_eval.cpu().numpy()  
            
            for v in range(graph_num):
                r_S = cal_homo_ratio(Scores[v].detach().cpu().numpy(), y_eval_all, self_loop=True)
                r_A = cal_homo_ratio(adjs_labels[v].detach().cpu().numpy(), y_eval_all, self_loop=True)
                ws[v] = r_A 
                
                if epoch_num > T0:
                    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
                    loss_contrastive = 0.0# torch.tensor(0.0, requires_grad=True)
                    contrastive_total_loss = 0.0# torch.tensor(0.0, requires_grad=True)
                    cluster_centers = cluster_centers_eval.to(device)   
                    cluster_centers = torch.tensor(cluster_centers, requires_grad=True)
                    
                    for i in range(class_num):
                        class_mask = (y_pred == i)
                        class_samples = h_all[class_mask].to(device)  
                        # class_samples = torch.tensor(class_samples)
                        # positive = cluster_centers[i].to(device)  
                        nearest_features = knn(h_all, class_samples, 20)
                        neg_mask = torch.ones(class_num, dtype=torch.bool)
                        neg_mask[i] = False
                        negative_centers = cluster_centers[neg_mask]  

                        for i, sample in enumerate(class_samples):  
                        # for sample in class_samples:
                            # for negative in negative_centers:
                                # loss1 = triplet_loss(sample.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                                # contrastive_total_loss += loss1.item()
                            
                            # contrastive_total_loss += batch_contrastive_loss(sample.unsqueeze(0), positive.unsqueeze(0).unsqueeze(0), negative_centers.unsqueeze(0))
                            contrastive_total_loss = contrastive_total_loss + batch_contrastive_loss(sample.unsqueeze(0), nearest_features[i].unsqueeze(0), negative_centers.unsqueeze(0)) / (len(h_all) * (class_num - 1))
                    loss_contrastive = loss_contrastive + contrastive_total_loss

            pgh = model.target_distribution(qgs[-1])
            loss_kl_g += F.kl_div(qgs[-1].log(), pgh, reduction='batchmean')
            for v in range(graph_num):
                loss_re_x += F.binary_cross_entropy(x_preds[v], xs[v])
                pg = model.target_distribution(qgs[v])
                loss_kl_g += F.kl_div(qgs[v].log(), pg, reduction='batchmean')
                loss_kl_g += F.kl_div(qgs[v].log(), pgh, reduction='batchmean')
            
            if l < kl_max:
                l = kl_step * epoch_num
            else:
                l = kl_max
            loss_kl_g *= l

            for v in range(graph_num):
                # output_sigmoid_Xs_recovery = torch.sigmoid(Xs_recovery[v]) 
                # output_sigmoid_xs = torch.sigmoid(xs[v]) 
                loss_Xs_recovery += F.mse_loss(Xs_recovery[v], xs[v])  
                # loss_Xs_recovery += F.mse_loss(output_sigmoid_Xs_recovery, output_sigmoid_xs) 
                # loss_Xs_recovery += F.binary_cross_entropy(output_sigmoid_Xs_recovery, output_sigmoid_xs)
                # loss_Zs_recovery += F.binary_cross_entropy(output_sigmoid_Zs_recovery, output_sigmoid_Zs)  
                # loss_hs_recovery += F.binary_cross_entropy(output_sigmoid_Zs_recovery, output_sigmoid_hs)  

                # loss_Xs_recovery += batch_contrastive_lossv1(Xs_recovery[v],xs[v],[item for index, item in enumerate(xs) if index != v])
                # loss_Xs_recovery += batch_contrastive_lossv2(Xs_recovery[v],xs[v],5)


                # S1 = Xs_recovery[v] @ Xs_recovery[v].T
                # S = xs[v] @ xs[v].T
                # loss_Xs_recovery += F.mse_loss(S1, S)

                # S1 = output_sigmoid_Xs_recovery @ output_sigmoid_Xs_recovery.T
                # S = output_sigmoid_xs @ output_sigmoid_xs.T
                # loss_Xs_recovery += F.mse_loss(S1, S)
                # c1 = F.normalize(Xs_recovery[v], dim=1)
                # c2 = F.normalize(xs[v], dim=1)
                # S1 = c1 @ c1.T
                # S = c2 @ c2.T
                # loss_Xs_recovery += F.mse_loss(S1, S)
                # c1 = F.normalize(output_sigmoid_Xs_recovery, dim=1)
                # c2 = F.normalize(output_sigmoid_xs, dim=1)
                # S1 = c1 @ c1.T
                # S = c2 @ c2.T
                # loss_Xs_recovery += F.mse_loss(S1, S)            
            loss += 1 * loss_re_x + 0 * loss_kl_g + 1 * loss_Xs_recovery + 0 * loss_Zs_recovery + 0 * loss_hs_recovery
            
            if epoch_num > T0:
                loss += 1 * loss_contrastive
            
            losses.append(loss.item())  
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            # =========================================evaluation=============================================================
            if epoch_num % update_interval == 0:
                model.eval()
                with torch.no_grad(): 
                    x_preds_eval, _, _, _, _, _, hs_eval, h_all_eval, _, _, _, _, _, _ = model(xs, adjs_labels, weights_h, noise_mode, dataset, ws)

                with torch.no_grad():
                    cluster_ids_eval_loop, _ = run_kmeans( # Using run_kmeans
                        h_all_eval, 
                        n_clusters=class_num
                    )
                    y_eval = cluster_ids_eval_loop 
                    
                nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz', visible=False)
                if acc > max_acc_overall:
                    max_acc_overall = acc
                if nmi > max_nmi_overall:
                    max_nmi_overall = nmi
                if ari > max_ari_overall:
                    max_ari_overall = ari
                if f1 > max_f1_overall:
                    max_f1_overall = f1

                current_sum_metrics = acc + nmi + ari + f1
                if current_sum_metrics > best_sum_metrics_val:
                    previous_acc_for_filename = acc_at_best_sum
                    
                    best_sum_metrics_val = current_sum_metrics
                    acc_at_best_sum = acc
                    nmi_at_best_sum = nmi
                    ari_at_best_sum = ari
                    f1_at_best_sum = f1
                    epoch_at_best_sum = epoch_num
                    bad_count = 0
                    
                    if previous_acc_for_filename > 1e-12: 
                        old_model_path = './pkl/NGCE_{}_acc{:.4f}.pkl'.format(dataset, previous_acc_for_filename)
                        if os.path.exists(old_model_path):
                            os.remove(old_model_path)
                    
                    torch.save({'state_dict': model.state_dict(),
                                'weights_h': weights_h,
                                'pseudo_label': pseudo_label, # pseudo_label from training step
                                'w': ws},
                               './pkl/NGCE_{}_acc{:.4f}.pkl'.format(dataset, acc_at_best_sum))
                    print('Saving model. ACC: {:.3f}, NMI: {:.3f}, ARI: {:.3f}, F1: {:.3f}, Ep: {}'.format(
                                        acc_at_best_sum, nmi_at_best_sum, ari_at_best_sum, f1_at_best_sum, epoch_at_best_sum))
                else:
                    bad_count += 1
                    current_loss_re_x_val = loss_re_x.item() if isinstance(loss_re_x, torch.Tensor) else loss_re_x
                    current_loss_xs_rec_val = loss_Xs_recovery.item() if isinstance(loss_Xs_recovery, torch.Tensor) else loss_Xs_recovery
                    

                if bad_count >= patience:
                    print('Early stopping. Training complete.')
                    print('Final Result: (ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, Sum: {:.4f}) achieved at epoch: {}'.format(
                        acc_at_best_sum, nmi_at_best_sum, ari_at_best_sum, f1_at_best_sum, best_sum_metrics_val, epoch_at_best_sum))
                    print()
                    break
        losses_array = np.array(losses)
        np.save('./npy/losses_values_{}'.format(dataset), losses_array)

        # print("Overall maximum metrics achieved during training:")
        # print(f"Max ACC: {max_acc_overall:.4f}")
        # print(f"Max NMI: {max_nmi_overall:.4f}")
        # print(f"Max ARI: {max_ari_overall:.4f}")
        # print(f"Max F1: {max_f1_overall:.4f}")

        n = 0 
        columns = ['dataset', 'acc', 'nmi', 'ari', 'f1', 'epoch', 'lr', 'weight_decay', 'order', 'hidden_dim', 'latent_dim', 'w', 'noise']

        dt = np.asarray(
            [dataset, acc_at_best_sum, nmi_at_best_sum, ari_at_best_sum, f1_at_best_sum, epoch_at_best_sum, lr, weight_decay, order, hidden_dim, latent_dim, w, 'yx']
        ).reshape(1, -1)
        df = pd.DataFrame(dt, columns=columns)
        # Check if file exists to determine header, n=0 logic might be per 'matadata' execution
        file_exists = os.path.exists('./result.csv')
        if n == 0 and not file_exists: # Write header if n=0 (first dataset in this run) AND file doesn't exist
             head = True
        else:
             head = False
        df.to_csv('./result.csv', index=False, header=head, mode='a')
        n += 1 

    if not train: 
        print("Warning, not training.")
    else: 
        if acc_at_best_sum > 1e-12 : 
            model_name = 'NGCE_{}_acc{:.4f}'.format(dataset, acc_at_best_sum)
        else: 
            print("Warning: No best model seemed to be saved during training based on acc_at_best_sum. Loading may fail or use a default.")
            model_name = 'NGCE_{}_acc{:.4f}'.format(dataset, 0.0) 
			
    print('Test complete...')