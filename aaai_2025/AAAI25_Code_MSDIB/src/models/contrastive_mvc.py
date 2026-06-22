import torch as th
import torch.nn as nn
import numpy as np
import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, MLP
from torch.nn.functional import softplus

from lib.fusion import get_fusion_module
from models.clustering_module import DDC
from models.model_base import ModelBase
from typing import Tuple, List, Union, Optional
from torch.distributions import Normal, Independent
from torch.nn.functional import normalize


class MSDIB(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the MSDIB model.

        :param cfg: Model config. See `config.defaults.CoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = \
            self.backbone_ddc_fused = self.backbone_ddc01 = self.backbone_ddc02 = self.v1 = self.v2 = None

        # Define Backbones and Fusion modules

        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)

        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"MSDIB requires all backbones to have the same " \
                                                          f"output size. Got: {bb_sizes}"

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])


        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())


    def forward(self, views):
        self.backbone_outputs = self.backbones(views)

        self.mu1, self.sigma1 = self.backbone_outputs[0][:,:128], self.backbone_outputs[0][:, 128:]
        self.mu2, self.sigma2 = self.backbone_outputs[1][:,:128], self.backbone_outputs[1][:, 128:]

        self.sigma1 = softplus(self.sigma1) + 1e-7  # Make sigma always positive
        self.sigma2 = softplus(self.sigma2) + 1e-7  # Make sigma always positive

        self.x_P_1 = Independent(Normal(loc=self.mu1, scale=self.sigma1), 1)
        self.x_P_2 = Independent(Normal(loc=self.mu2, scale=self.sigma2), 1)

        self.fused = self.fusion( self.backbone_outputs)
        self.mu_f, self.sigma_f = self.fused[:,:128], self.fused[:, 128:]
        self.sigma_f = softplus(self.sigma_f) + 1e-7  # Make sigma always positive
        self.x_P_f = Independent(Normal(loc=self.mu_f, scale=self.sigma_f), 1)

        self.backbone_ddc01, _ = self.ddc(self.backbone_outputs[0])
        self.backbone_ddc02, _ = self.ddc(self.backbone_outputs[1])
        self.output, self.hidden = self.ddc(self.fused)

        return self.output, self.backbone_ddc01, self.backbone_ddc02

