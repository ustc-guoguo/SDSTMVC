import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
from alignment import *
def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x

def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx


def pretrain(model, optimizer, pretrain_loader, criterion, args):
    print('pretraining ...')
    model.train()
    losses = AverageMeter()
    t_progress = tqdm(range(args.prebs+args.Cbs), desc='Pretraining')
    for epoch in t_progress:
        current_loss = 0
        count = 0
        for batch_idx, (x0, x1) in enumerate(pretrain_loader):
            x0, x1 = x0.to(args.gpu), x1.to(args.gpu)
            print(np.shape(x0.view(x0.size()[0], -1)),'s')
            try:
                h0, h1, d0, d1 = model(x0.view(x0.size()[0], -1), x1.view(x1.size()[0], -1))
            except Exception as e:
                print("error raise in batch", batch_idx, "-", repr(e))
                continue
            # S0,S1=cosineSimilarty(h0,h0),cosineSimilarty(h1,h1)
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
            loss1_1 = criterion(x0, d0)
            loss1_2 = criterion(x1, d1)
            loss1=loss1_1+loss1_2
            loss2=criterion(h0,h1)
            loss=loss1+loss2
            if (epoch >= args.prebs):  ##ae_epochs: 2000
                n_samples = h0.shape[0]
                P_index = random.sample(range(n_samples), n_samples)
                P_gt = np.eye(n_samples).astype('float32')
                P_gt = P_gt[:, P_index]
                P_gt = torch.from_numpy(P_gt).to(args.gpu)
                h1=torch.mm(P_gt,h1)
                C = cosineSimilartydis(h0, h1)
                P_pred = alignment(C)
                loss += args.lamda * F.mse_loss(P_pred, P_gt.T)  ##batch_P是地面真实对齐
                # optimizer.param_groups[0]["lr"]=0.002
            if epoch != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.update(loss.item())
            current_loss += loss.item()
            count += 1
        t_progress.write('epoch %d : loss %.6f' % (epoch, current_loss / count))
        t_progress.set_description_str(' Loss=' + str(losses.avg))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def row2one(P):
    P_sum = P.sum(dim=1, keepdim=True)
    one = torch.ones(1, P.shape[1]).to(P.device)
    return P - (P_sum - 1).mm(one) / P.shape[1]

def col2one(P):
    P_sum = P.sum(dim=0, keepdim=True)
    one = torch.ones(P.shape[0], 1).to(P.device)
    return P - (one).mm(P_sum - 1) / P.shape[0]

def P_init(D):
    P = torch.zeros_like(D)
    D_rowmin = D.clone()
    max_d = D.max()
    min_ind = torch.argmin(D_rowmin, dim=0)
    D_rowmin[:, :] = max_d
    D_rowmin = D_rowmin.scatter(0, min_ind.unsqueeze(0), D[min_ind, torch.arange(D.shape[1]).long()].unsqueeze(0))

    _, idx_max = torch.min(D_rowmin, dim=1)

    P[torch.arange(D.shape[0]).long(), idx_max.long()] = 1.0

    return P
def alignment(D, tau_1=30, tau_2=10, lr=0.1):
    P = P_init(D)

    d = [torch.zeros_like(D) for _ in range(3)]

    for i in range(tau_1):
        P = P - lr * D
        for j in range(tau_2):
            P_0 = P.clone()
            P = P + d[0]
            Y = row2one(P)
            d[0] = P - Y

            P = Y + d[1]
            Y = col2one(P)
            d[1] = P - Y

            P = Y + d[2]
            Y = F.relu(P)
            d[2] = P - Y

            P = Y
            if (P - P_0).norm().item() == 0:
                break

    return P