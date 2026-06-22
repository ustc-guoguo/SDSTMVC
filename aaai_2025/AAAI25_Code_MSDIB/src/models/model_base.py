import torch.nn as nn
import torch as th
import numpy as np

IGNORE_IN_TOTAL = ("",)
device = th.device("cuda" if th.cuda.is_available() else "cpu")


class ModelBase(nn.Module):
    def __init__(self):
        """
        Model base class
        """
        super().__init__()

        self.fusion = None
        self.optimizer = None
        self.loss = None

    def calc_losses(self, ignore_in_total=tuple()):
        return self.loss(self, ignore_in_total=ignore_in_total)

    def train_step(self, batch, epoch, it, n_batches):
        self.optimizer.zero_grad()
        _, _, _ = self(batch)
        losses = self.calc_losses()
        losses["tot"].backward()
        self.optimizer.step(epoch + it / n_batches)
        return losses





