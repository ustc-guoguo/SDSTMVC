import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyEnhancedContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=1.0, use_energy_weighting=True, device='cuda'):
        """
        Energy-based contrastive loss function (using the same augmentation strategy for positive and negative samples), supports energy weighting
        Args:
            batch_size: Size of batch
            temperature: Temperature parameter used to scale similarity
            use_energy_weighting: Whether to use energy to calculate view importance
            device: Device information
        """
        super(EnergyEnhancedContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # Define energy_scale as a learnable parameter
        # self.energy_scale = nn.Parameter(torch.tensor(energy_scale, dtype=torch.float32, device=device))
        self.use_energy_weighting = use_energy_weighting

    def compute_energy_importance(self, E_H, E_r):
        """
        Calculate view importance based on energy
        Args:
            E_H: Energy of global features [batch_size, 1]
            E_r: Energy of single view features [batch_size, 1]
        Returns:
            importance_weight: View importance weight (scalar)
        """
        # Calculate average energy
        avg_E_H = E_H.mean()
        avg_E_r = E_r.mean()

        # Calculate relative energy difference
        energy_diff = torch.abs(avg_E_H - avg_E_r)

        # The closer the energy, the more important the view (smaller energy difference, larger weight)
        importance = torch.exp(-energy_diff)

        # Limit the weight range to avoid weights being too large or too small
        importance = torch.clamp(importance, min=0.1, max=2.0)

        return importance.item()

    def forward(self, H, r, model=None, energy_scale=0.1):
        """
        Calculate energy-based contrastive loss (using the same augmentation strategy for positive and negative samples)
        Args:
            H: Global representation [batch_size, feature_dim]
            r: Features from a single view [batch_size, feature_dim]
            model: Multi-view clustering model, includes energy network
            energy_scale: Scaling coefficient for energy enhancement term
        """
        # Check if input tensors are valid
        if H is None or r is None or H.size(0) == 0 or r.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        batch_size = r.size(0)
        device = r.device

        # View energy importance weight
        energy_importance = 1.0

        # 1. Basic similarity calculation
        sim_matrix = torch.matmul(r, H.t()) / self.temperature

        # 2. If model is provided, add energy enhancement term for all samples (positive and negative)
        if model is not None:
            # Calculate energy for H and r
            E_H = model.energy_net(H)  # [batch_size, 1]
            E_r = model.energy_net(r)  # [batch_size, 1]

            # If energy weighting is enabled, calculate view importance
            if self.use_energy_weighting:
                energy_importance = self.compute_energy_importance(E_H, E_r)

            # Calculate energy difference between all sample pairs
            E_r_expanded = E_r.expand(batch_size, batch_size)
            E_H_expanded = E_H.transpose(0, 1).expand(batch_size, batch_size)
            energy_diff = -torch.abs(E_r_expanded - E_H_expanded)

            # Apply energy difference enhancement to all samples
            energy_similarity = energy_diff

            # Apply energy enhancement to similarity matrix
            energy_similarity = torch.clamp(energy_similarity, min=-5.0, max=5.0)
            enhanced_sim_matrix = sim_matrix + energy_scale * energy_similarity
        else:
            enhanced_sim_matrix = sim_matrix

        # Extract positive sample similarity (diagonal elements)
        pos_sim = torch.diag(enhanced_sim_matrix).view(batch_size, 1)
        # Create mask to exclude diagonal (self)
        mask = torch.eye(batch_size, device=device) == 0
        neg_sim = enhanced_sim_matrix[mask].view(batch_size, -1)
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Get loss
        base_loss = self.criterion(logits, labels)

        # Apply view importance and global weight
        loss = energy_importance * base_loss

        return loss