import sys
import torch
import torch.nn as nn

EPS = sys.float_info.epsilon


def mutual_information(x_img, x_txt):
    _, k = x_img.size()
    p_i_j = compute_joint(x_img, x_txt)
    assert (p_i_j.size() == (k, k))
    a = p_i_j.sum(dim=1)
    temp1 = p_i_j.sum(dim=1).view(k, 1)
    p_i = temp1.expand(k, k).clone()
    temp2 = p_i_j.sum(dim=0).view(1, k)
    p_j = temp2.expand(k, k).clone()
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss


def compute_joint(x_img, x_txt):
    bn, k = x_img.size()
    assert (x_txt.size(0) == bn and x_txt.size(1) == k)
    b = x_img.unsqueeze(2)
    c = x_txt.unsqueeze(1)
    p_i_j = x_img.unsqueeze(2) * x_txt.unsqueeze(1)
    d = p_i_j.sum(dim=0)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    p_i_j = p_i_j / p_i_j.sum()
    return p_i_j