import torch as th
import torch.nn as nn
import numpy as np
import config
from lib import kernel
import math
import torch.distributions.normal as normal
import torch.nn.functional as F
from lib.mutual_information import mutual_information
from models.model import MIEstimator
from models.model import UD_constraint_f



EPSILON = 1E-9
DEBUG_MODE = False
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, cfg, extra):
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def __call__(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def __call__(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def __call__(self, net, cfg, extra):
        m = th.exp(-kernel.cdist(net.output, self.eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)


prior_loc = th.zeros(100, 128)
prior_scale = th.ones(100, 128)
prior = normal.Normal(prior_loc, prior_scale)
mi_estimator = MIEstimator(128, 128).to(device)
def getMILoss(P_F, P, mi_estimator):
    x_P_F = P_F.rsample()
    prior_sample = prior.sample().to(device)
    loss1_1 = 0
    loss2 = 0
    loss3 = 0
    for p in P:
        x_p = p.rsample()
        miG, _ = mi_estimator(x_P_F, x_p)
        loss2 += mutual_information(x_p, x_P_F)
        loss1_1 += -miG
        loss3 += th.nn.functional.kl_div(x_p, prior_sample)
    return loss1_1, loss2, loss3

class MASG(LossTerm):

    def __init__(self, cfg):
        super().__init__()

    def __call__(self, net, cfg, extra):
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        x1_cluster = th.softmax(net.backbone_ddc01, dim=1)
        x2_cluster = th.softmax(net.backbone_ddc02, dim=1)

        x3_cluster = th.softmax(net.output, dim=1)
        lossKL, lossMI, _ = getMILoss(net.x_P_f, [net.x_P_1, net.x_P_2], mi_estimator)
        loss1 = lossKL+10*lossMI

        loss2 = mutual_information(net.backbone_outputs[0], net.backbone_outputs[1])
        loss3 = (mutual_information(net.backbone_outputs[0], net.fused)+
                 mutual_information(net.backbone_outputs[1], net.fused))
        loss4 = mutual_information(net.backbone_ddc01, net.output) + mutual_information(net.backbone_ddc02, net.output)

        with th.no_grad():
            UDC_img1 = UD_constraint_f(x1_cluster).to(device)
            UDC_txt1 = UD_constraint_f(x2_cluster).to(device)
            UDC_txt3 = UD_constraint_f(x3_cluster).to(device)
        criterion_cross = th.nn.CrossEntropyLoss().to(device)
        loss_op = criterion_cross(x1_cluster, UDC_img1) + criterion_cross(x2_cluster, UDC_txt1)+criterion_cross(x3_cluster,UDC_txt3)
        loss_1= cfg.c1*(0.01*loss1+loss2)
        loss_2 = cfg.c2*(loss4+loss3+loss_op)
        loss =loss_2-loss_1
        return loss




# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)


# ======================================================================================================================
# Loss class
# ======================================================================================================================


class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "MASG": MASG,
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            # if term_name != 'contrast' and term_name != 'instance_cluster':
            self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])

        return loss_values




