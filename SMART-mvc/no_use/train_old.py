import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.io as sio
from evaluation import eval_aligned_detail, evaluate
from clustering import clustering
# from calculate_graph import calculate_graphs, calculate_laplacian, calculate_cosine_similarity, knn
from loss import crossview_contrastive_Loss
from stochastic import wasserstein_initialisation, regularise_and_invert
from utils import *

def pretrain(model, optimizer, config, x1_train, x2_train, flag, Y_list, logger, pretrain_path, device):
    for epoch in range(config['training']['pre_epoch']):
        loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre = 0, 0, 0, 0, 0
        for batch_x1_aligned, batch_x2_aligned, batch_x1_mis_aligned, batch_x2_mis_aligned, batch_No in next_batch_aligned(
                x1_train, x2_train, flag, config['training']['batch_size']):
            z1 = model.autoencoder1.encoder(batch_x1_aligned)
            z2 = model.autoencoder2.encoder(batch_x2_aligned)
            z1_mis = model.autoencoder1.encoder(batch_x1_mis_aligned)
            z2_mis = model.autoencoder2.encoder(batch_x2_mis_aligned)
            # Within-view Reconstruction Loss
            recon1 = F.mse_loss(model.autoencoder1.decoder(z1), batch_x1_aligned)
            recon2 = F.mse_loss(model.autoencoder2.decoder(z2), batch_x2_aligned)
            recon3 = F.mse_loss(model.autoencoder1.decoder(z1_mis), batch_x1_mis_aligned)
            recon4 = F.mse_loss(model.autoencoder2.decoder(z2_mis), batch_x2_mis_aligned)
            reconstruction_loss = recon1 + recon2 + recon3 + recon4

            # Cross-view Contrastive_Loss
            cl_loss = crossview_contrastive_Loss(z1, z2, config['training']['alpha'])

            # Cross-view Dual-Prediction Loss
            z1_hat, _ = model.generator1(z1)
            z2_hat, _ = model.generator2(z2)
            pre1 = F.mse_loss(z1_hat, z2, reduction='sum')
            pre2 = F.mse_loss(z2_hat, z1, reduction='sum')
            dualprediction_loss = (pre1 + pre2)

            loss = cl_loss + reconstruction_loss * config['training']['lambda2']

            # we train the autoencoder by L_cl and L_rec first to stabilize
            # the training of the dual prediction
            if epoch >= config['training']['start_dual_prediction']:
                loss += dualprediction_loss * config['training']['lambda1']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            loss_rec1 += recon1.item()
            loss_rec2 += recon2.item()
            loss_pre += dualprediction_loss.item()
            loss_cl += cl_loss.item()

        if (epoch + 1) % config['print_num'] == 0:
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                .format((epoch + 1), config['training']['epoch'], loss_rec1, loss_rec2, loss_pre, loss_cl, loss_all)

            logger.info("\033[2;29m" + output + "\033[0m")

        if (epoch + 1) % config['print_num'] == 0:
            scores = model.evaluation(logger, x1_train, x2_train, Y_list)
    torch.save(model.state_dict(), pretrain_path)


def train(model, optimizer, config, X1, X2, flag, label_list, logger, device):

    metrics_list = {'acc': [], 'nmi': [], 'ari': [], 'pur': []}
    metrics_list_1 = {'acc_1': [], 'nmi_1': [], 'ari_1': [], 'pur_1': []}
    metrics_list_2 = {'acc_2': [], 'nmi_2': [], 'ari_2': [], 'pur_2': []}
    loss_list = {'loss': [], 'rec_loss': [], 'intra_loss': [], 'inter_loss': []}
    acc_best, nmi_best, ari_best, pur_best, best_epoch = 0.0, 0.0, 0.0, 0.0, 0
    acc_1_best, nmi_1_best, ari_1_best, pur_1_best, best_1_epoch = 0.0, 0.0, 0.0, 0.0, 0
    acc_2_best, nmi_2_best, ari_2_best, pur_2_best, best_2_epoch = 0.0, 0.0, 0.0, 0.0, 0

    # training
    for epoch in range(config['epochs']):
        # Forward
        model.train()
        emb1, emb2, X1_hat, X2_hat = model(X1, X2)
        # Within-view Reconstruction Loss
        loss_rec = model.reconstruction_loss(X1=X1,
                                             X2=X2,
                                             X1_hat=X1_hat,
                                             X2_hat=X2_hat)
        # if epoch < 100:
        #     loss = loss_rec
        #     Z1_p1, Z2_p1, _, _ = model.project(emb1, emb2)
        #     model.inter_sim_graph(Z1_p1, Z2_p1, threshold=config['threshold2'])
        # else:
        Z1_p1, Z2_p1, Z1_p2, Z2_p2 = model.project(emb1, emb2)
        # Z1_p1, Z2_p1 = model.graph_projector(emb1, emb2)
        # Z1_p2, Z2_p2 = model.clustering_projector(emb1, emb2)

        loss_intra_NCL = model.intra_modal_NCL(Z1_p1, Z1_p2, threshold=config['threshold1'], tau=config['tau'])
        loss_intra_NCL += model.intra_modal_NCL(Z2_p1, Z2_p2, threshold=config['threshold1'], tau=config['tau'])

        loss_inter_NCL = model.inter_modal_NCL(emb1, emb2, Z1_p1, Z2_p1, threshold=config['threshold2'], tau=config['tau'])

        # if  config['lambda1'] != 0 and config['lambda2'] != 0:  # 11
        #     loss = config['lambda1'] * loss_intra_NCL + config['lambda2'] * loss_inter_NCL
        # elif config['lambda1'] != 0 and config['lambda2'] == 0:  # 10
        #     loss = config['lambda1'] * loss_intra_NCL
        # elif config['lambda1'] == 0 and config['lambda2'] != 0:  # 01
        #     loss = config['lambda2'] * loss_inter_NCL

        # loss = loss_rec + config['lambda1'] * loss_intra_NCL + config['lambda2'] * loss_inter_NCL
        if config['lambda0'] != 0 and config['lambda1'] != 0 and config['lambda2'] != 0:  # 111
            loss = config['lambda0'] * loss_rec + config['lambda1'] * loss_intra_NCL + config['lambda2'] * loss_inter_NCL
        elif config['lambda0'] != 0 and config['lambda1'] != 0 and config['lambda2'] == 0:  # 110
            loss = config['lambda0'] * loss_rec + config['lambda1'] * loss_intra_NCL
        elif config['lambda0'] != 0 and config['lambda1'] == 0 and config['lambda2'] != 0:  # 101
            loss = config['lambda0'] * loss_rec + config['lambda2'] * loss_inter_NCL
        elif config['lambda0'] == 0 and config['lambda1'] != 0 and config['lambda2'] != 0:  # 011
            loss = config['lambda1'] * loss_intra_NCL + config['lambda2'] * loss_inter_NCL
        elif config['lambda0'] != 0 and config['lambda1'] == 0 and config['lambda2'] == 0:  # 100
            loss = config['lambda0'] * loss_rec
        elif config['lambda0'] == 0 and config['lambda1'] != 0 and config['lambda2'] == 0:  # 010
            loss = config['lambda1'] * loss_intra_NCL
        elif config['lambda0'] == 0 and config['lambda1'] == 0 and config['lambda2'] != 0:  # 001
            loss = config['lambda2'] * loss_inter_NCL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list['loss'].append(loss.item())
        # loss_list['rec_loss'].append(loss_rec.item())
        # loss_list['intra_loss'].append(loss_intra_NCL.item())
        # loss_list['inter_loss'].append(loss_inter_NCL.item())

        model.eval()
        emb1, emb2 = model(X1, X2, is_eval=True)
        # emb1, emb2 = model.clustering_projector(emb1, emb2)
        # Z2 = torch.mm(model.adj, Z2.detach())
        # P = model.adj_inter / torch.norm(model.adj_inter, dim=-1, keepdim=True)
        # P = (model.adj_inter == model.adj_inter.max(dim=-1, keepdim=True)[0]).float()

        # fig_path = config['save_path'] + f"heatmap/{config['dataset']}/"
        # if not os.path.exists(fig_path):
        #     os.makedirs(fig_path)
        # show_heat_map(model.adj_inter.detach().cpu(),
        #               label=label_true,
        #               P_index=P_index,
        #               title="adj",
        #               save_path=f"{fig_path}/adj")
        # show_heat_map(P.detach().cpu(),
        #               label=label_true,
        #               P_index=P_index,
        #               title="P",
        #               save_path=f"{fig_path}/P")

        emb2_hat = torch.matmul(model.adj_inter, emb2).detach()
        emb2_hat = F.normalize(emb2_hat, p=2, dim=-1)
        # emb2 = torch.matmul(min_max_normalize(model.adj_inter), emb2)
        emb_fusion = get_fusion(Z1=emb1.detach(),
                              Z2=emb2_hat,
                              ratio=config['fusion_ratio'],
                              fusion_mode=config['fusion_mode'])

        label_1_pred, _, _ = clustering(feature=emb1.detach(),
                                      cluster_num=config['n_classes'],
                                      device=device)
        label_pred, _, _ = clustering(feature=emb_fusion,
                                      cluster_num=config['n_classes'],
                                      device=device)
        label_2_pred, _, _ = clustering(feature=emb2.detach(),
                                        cluster_num=config['n_classes'],
                                        device=device)
        label_true1 = label_list[0]
        label_true2 = label_list[1]
        acc, nmi, ari, pur = evaluate(label_true1, label_pred)
        acc_1, nmi_1, ari_1, pur_1 = evaluate(label_true1, label_1_pred)
        acc_2, nmi_2, ari_2, pur_2 = evaluate(label_true2, label_2_pred)
        metrics_list['acc'].append(acc)
        metrics_list['nmi'].append(nmi)
        metrics_list['ari'].append(ari)
        metrics_list['pur'].append(pur)
        metrics_list_1['acc_1'].append(acc_1)
        metrics_list_1['nmi_1'].append(nmi_1)
        metrics_list_1['ari_1'].append(ari_1)
        metrics_list_1['pur_1'].append(pur_1)
        metrics_list_2['acc_2'].append(acc_2)
        metrics_list_2['nmi_2'].append(nmi_2)
        metrics_list_2['ari_2'].append(ari_2)
        metrics_list_2['pur_2'].append(pur_2)
        is_best = False
        score = [acc, nmi, ari, pur, is_best]
        acc_best, nmi_best, ari_best, pur_best, is_best = compare_score(score,
                                                                      (acc_best, nmi_best, ari_best, pur_best))
        if is_best:
            best_epoch = epoch+1
        acc_1_best, nmi_1_best, ari_1_best, pur_1_best, is_best = compare_score((acc_1, nmi_1, ari_1, pur_1, is_best),
                                                                        (acc_1_best, nmi_1_best, ari_1_best, pur_1_best))
        if is_best:
            best_1_epoch = epoch+1
        acc_2_best, nmi_2_best, ari_2_best, pur_2_best, is_best = compare_score((acc_2, nmi_2, ari_2, pur_2, is_best),
                                                                        (acc_2_best, nmi_2_best, ari_2_best, pur_2_best))
        if is_best:
            best_2_epoch = epoch+1
        # if acc >= acc_best:
        #     acc_best = acc
        #     nmi_best = nmi
        #     ari_best = ari
        #     f1_best = f1
        #     best_epoch = epoch + 1
        # if acc_1 >= acc_1_best:
        #     acc_1_best = acc_1
        #     nmi_1_best = nmi_1
        #     ari_1_best = ari_1
        #     f1_1_best = f1_1
        #     best_1_epoch = epoch + 1
        # if acc_2 >= acc_2_best:
        #     acc_2_best = acc_2
        #     nmi_2_best = nmi_2
        #     ari_2_best = ari_2
        #     f1_2_best = f1_2
        #     best_2_epoch = epoch + 1

        if epoch == 0 or (epoch + 1) % 20 == 0:
            # if epoch < 100:
            #     logger.info(
            #         f"[Epoch {epoch + 1:<3}] loss: {loss.item():.8f}, rec_loss: {loss_rec.item():.8f}")
            #     logger.info(f'      ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}')
            # else:
            logger.info(
                f"[Epoch {epoch + 1:<3}] loss: {loss.item():.8f}, rec_loss: {loss_rec.item():.8f}, intra_loss: {loss_intra_NCL.item():.8f}, inter_loss: {loss_inter_NCL.item():.8f}")
            logger.info(f'      ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, PUR: {pur:.4f}')
            # logger.info(f'  V1: ACC: {acc_1:.4f}, NMI: {nmi_1:.4f}, ARI: {ari_1:.4f}, F1: {f1_1:.4f}')
            # logger.info(f'  V2: ACC: {acc_2:.4f}, NMI: {nmi_2:.4f}, ARI: {ari_2:.4f}, F1: {f1_2:.4f}')

    logger.info(
        f'Best: Epoch {best_epoch}, ACC {acc_best:.4f}, NMI {nmi_best:.4f}, ARI {ari_best:.4f}, PUR {pur_best:.4f}')
    logger.info(
        f'Best V1: Epoch {best_1_epoch}, ACC {acc_1_best:.4f}, NMI {nmi_1_best:.4f}, ARI {ari_1_best:.4f}, PUR {pur_1_best:.4f}')
    logger.info(
        f'Best V2: Epoch {best_2_epoch}, ACC {acc_2_best:.4f}, NMI {nmi_2_best:.4f}, ARI {ari_2_best:.4f}, PUR {pur_2_best:.4f}')
    best = (acc_best, nmi_best, ari_best, pur_best)
    best_v1 = (acc_1_best, nmi_1_best, ari_1_best, pur_1_best)
    best_v2 = (acc_2_best, nmi_2_best, ari_2_best, pur_2_best)

    return best, best_v1, best_v2


def train0(model, optimizer, config, x1_train, x2_train, flag, Y_list, index_mis_aligned, P_gt, logger, device):
    # 指标
    best_acc = 0
    acc, nmi, ari = [], [], []

    # got config
    epochs = range(config['epoch'])
    got_init_epoch = config['training']['got']['init_epoch']
    got_update_epoch = config['training']['got']['update_epoch']

    # init got
    fea1, fea2 = get_cat_feature(model, x1_train, x2_train)
    fea1 = to_numpy(fea1)
    fea2 = to_numpy(fea2)
    dim = int(fea1.shape[1] / 2)
    similarity = calculate_cosine_similarity(fea1[~flag], fea2[~flag])
    similarity = to_tensor(similarity, device)
    model.got.init_param(similarity)

    # training
    for epoch in epochs:
        # update P
        update = 50
        if epoch % 50 == 0:
            fea1, fea2 = get_cat_feature(model, x1_train, x2_train)

            g1, g2, L1_reg, L2_reg = get_got_input(fea1[~flag][:, dim:], fea2[~flag][:, :dim], config, k=config['k'])
            if epoch == 0:
                got_epoch = got_init_epoch
            else:
                got_epoch = got_update_epoch
            P, P_pred = train_got(model.got, L1_reg, L2_reg, optimizer, config, got_epoch, device, similarity.detach(), int(epoch / update))
            # eval got delete
            eval_aligned_detail(P, P_pred, index_mis_aligned, Y_list[0])
            P_global = get_global_p(flag, P, device=device)
        x1_recon, x2_recon, z1, z2, z1_hat, z2_hat = model(x1_train, x2_train)
        # Within-view Reconstruction Loss
        recon1 = F.mse_loss(x1_recon, x1_train)
        recon2 = F.mse_loss(x2_recon, x2_train)
        reconstruction_loss = recon1 + recon2

        # Cross-view Contrastive_Loss
        cl_loss = crossview_contrastive_Loss(z1[flag], z2[flag], config['alpha'])

        # Cross-view Dual-Prediction Loss
        pre1 = F.mse_loss(z1_hat, P_global @ z2)
        pre2 = F.mse_loss(P_global @ z2_hat, z1)
        dualprediction_loss = (pre1 + pre2)

        loss = cl_loss + reconstruction_loss * config['lambda2'] + dualprediction_loss * config['lambda1']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch == 0 or (epoch + 1) % config['print_num'] == 0:
        #     output = f"Epoch {epoch+1:>3}/{config['epoch']}: T_loss = {:.8f}, Rec loss = {:.8f} " \
        #              "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
        #         .format((epoch + 1), config['epoch'], recon1.item(), recon2.item(), dualprediction_loss.item(), cl_loss.item(), loss.item())
        #
        #     logger.info("\033[2;29m" + output + "\033[0m")

        if (epoch + 1) % 1 == 0:
            latent_fusion = torch.cat([P_global @ z2_hat.detach(), z1_hat.detach()], dim=1).cpu().numpy()
            # graph1, graph2 = calculate_graphs(fake_z2.detach(), fake_z1.detach())
            # graph2 = P_global @ graph2 @ P_global.t()
            # graph = (graph1 + graph2) * 0.5
            # # scores = clustering([latent_fusion], Y_list[0])
            # scores = post_clustering(graph.cpu().numpy(), Y_list[0], config['cluster_param'])
            scores = clustering([latent_fusion], Y_list[0])
            acc.append(scores['kmeans']['accuracy'])
            nmi.append(scores['kmeans']['NMI'])
            ari.append(scores['kmeans']['ARI'])
            if best_acc < scores['kmeans']['accuracy']:
                best_acc = scores['kmeans']['accuracy']
            logger.info("\033[2;29m" + 'epoch' + str(epoch) + '     ===>view_concat ' + str(scores) + "\033[0m")
    print('best_accuracy: %.4f' % best_acc)
    plt.plot(epochs, acc, 'y-', label='acc')
    plt.plot(epochs, nmi, 'b-', label='nmi')
    plt.legend(loc='upper right')
    plt.show()
    num = 5
    acc = np.array(acc)
    idx = np.argsort(-acc)
    nmi = np.array(nmi)
    ari = np.array(ari)

    best_acc_num = acc[idx[:num]].tolist()
    best_nmi_num = nmi[idx[:num]].tolist()
    best_ari_num = ari[idx[:num]].tolist()

    print('acc: %s' % (str(best_acc_num)))
    print('nmi: %s' % (str(best_nmi_num)))
    print('ari: %s' % (str(best_ari_num)))
    return best_acc_num, best_nmi_num, best_ari_num


def train_got(model, L1_reg, L2_reg, optimizer, config, epochs, device, similarity, item):
    # Initialization
    torch.manual_seed(config['training']['got']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['got']['seed'])

    L1_tensor = to_tensor(L1_reg, device)
    L2_tensor = to_tensor(L2_reg, device)
    params = wasserstein_initialisation(L1_reg, L2_reg)
    history = []
    for epoch in range(epochs):
        cost = 0
        for iter in range(config['training']['got']['num_iter']):
            eps = torch.randn((model._nodes, model._nodes)).to(device)
            DS = model(eps)
            loss = model.loss_got(L1_tensor, L2_tensor, DS, params)
            cost += loss
        cost = cost / config['training']['got']['num_iter']
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        history.append(cost.item())
        if epoch % 50 == 0:
            print('[Epoch %4d/%d] loss: %f - std: %f' % (epoch, epochs, cost.item(), model.std.detach().mean()))
    P = model.doubly_stochastic(model.mean)
    max_val, max_idx = torch.max(P, dim=1, keepdim=True)
    P_pred = torch.zeros_like(P)
    P_pred.scatter_(1, max_idx, 1)
    return P.detach(), P_pred.detach()


def get_cat_feature(model, x1_train, x2_train):
    with torch.no_grad():
        model.eval()
        # x1_recon, x2_recon, z1, z2, z1_hat, z2_hat
        x1_recon, x2_recon, feature1, feature2, fake_fea2, fake_fea1 = model(x1_train, x2_train)
        model.train()
    fea_cat1 = torch.cat((feature1, fake_fea2), dim=1)
    fea_cat2 = torch.cat((fake_fea1, feature2), dim=1)
    return fea_cat1, fea_cat2

def get_got_input(fea1, fea2, config, k=100, graph=True):
    # 1. g1 and g2
    g1, g2 = calculate_graphs(fea1, fea2)
    # 2. L1 and L2
    L1 = calculate_laplacian(g1, k=k).cpu().numpy()
    L2 = calculate_laplacian(g2, k=k).cpu().numpy()
    if graph:
        [L1_reg, L2_reg] = regularise_and_invert(L1, L2, config['training']['got']['alpha'], ones=True)
    else:
        L1_reg = L1
        L2_reg = L2
    return g1, g2, L1_reg, L2_reg

def get_global_p(flag, P_part, device='cuda'):
    num_sample = flag.shape[0]
    P = torch.zeros(num_sample, dtype=torch.float32).to(device)
    vec = torch.zeros((num_sample, )).to(device)
    vec[flag] = 1
    P = P + torch.diag(vec)
    idx = []
    for i in range(len(flag)):
        if flag[i] == False:
            idx.append(i)
    for i in range(P_part.shape[0]):
        P[idx[i], idx] = P_part[i]
    return P