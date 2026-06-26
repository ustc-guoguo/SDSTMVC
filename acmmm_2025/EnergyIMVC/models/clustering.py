import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from .networks import ViewAttention, ImputationNetwork, EnergyNetwork


class MultiViewClusteringModel(nn.Module):
    def __init__(self, view_dims, feature_dim=64, high_feature_dim=20,
                 margin=1.0, imputation_loss_weight=0.1, cd_loss_weight=0.1,
                 hidden_dim=256, dropout=0.1, num_heads=4):  # Add num_heads parameter
        super(MultiViewClusteringModel, self).__init__()
        self.view = len(view_dims)
        self.margin = margin
        self.imputation_loss_weight = imputation_loss_weight
        self.cd_loss_weight = cd_loss_weight
        self.num_heads = num_heads  # Save num_heads parameter
        self._init_networks(view_dims, feature_dim, high_feature_dim, hidden_dim, dropout)

    def _init_networks(self, view_dims, feature_dim, high_feature_dim, hidden_dim, dropout):
        # Encoders and decoders
        self.encoders = nn.ModuleList([
            self._create_encoder(dim, feature_dim, hidden_dim, dropout) for dim in view_dims
        ])
        self.decoders = nn.ModuleList([
            self._create_decoder(dim, feature_dim, hidden_dim, dropout) for dim in view_dims
        ])

        # Feature modules with normalization and dropout
        self.common_feature = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            nn.LayerNorm(high_feature_dim),
            nn.Dropout(dropout)
        )

        # Network components
        self.view_attention = ViewAttention(feature_dim=high_feature_dim, num_heads=self.num_heads)

        # Imputation network, input is high_feature_dim, i.e., r's dimension
        self.imputation_nets = nn.ModuleList([
            ImputationNetwork(input_dim=high_feature_dim, hidden_dim=hidden_dim) for _ in range(self.view)
        ])

        # Energy network, input is high_feature_dim, i.e., r's dimension
        self.energy_net = EnergyNetwork(input_dim=high_feature_dim)

    @staticmethod
    def _create_encoder(input_dim, output_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def _create_decoder(output_dim, input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def _process_views_with_mask(self, views, missing_info=None):
        """Process view data with mask, special handling for missing data"""
        xrs, zs, rs = [], [], []

        for v in range(self.view):
            if missing_info is not None:
                # 1 means missing, 0 means available
                mask = (1 - missing_info[:, v]).unsqueeze(1)

                # Process non-missing data
                x_available = views[v]
                z = self.encoders[v](x_available)
                xr = self.decoders[v](z)
                r = normalize(self.common_feature(z), dim=1)

                # Set missing data to zero (mask=0)
                z = z * mask
                xr = xr * mask
                r = r * mask
            else:
                # Normal processing for non-missing data
                z = self.encoders[v](views[v])
                xr = self.decoders[v](z)
                r = normalize(self.common_feature(z), dim=1)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        return xrs, zs, rs

    def _compute_view_energies(self, rs, missing_info=None):
        """
        Calculate average energy for each view
        Only consider non-missing samples for average energy calculation

        Args:
            rs: list of [batch_size, feature_dim] for each view
            missing_info: [batch_size, n_views], 1 means missing, 0 means available
        Returns:
            view_energies: [n_views] average energy for each view
        """
        n_views = len(rs)
        device = rs[0].device

        # Initialize view energies
        view_energies = torch.zeros(n_views, device=device)

        # Calculate average energy for each view
        for v in range(n_views):
            if missing_info is not None:
                # Get mask for non-missing samples
                available_mask = (missing_info[:, v] == 0)
                # If this view has available samples
                if available_mask.sum() > 0:
                    # Only calculate energy for non-missing samples
                    available_indices = torch.nonzero(available_mask).squeeze(1)
                    if len(available_indices) > 0:
                        available_features = rs[v][available_indices]
                        energies = self.energy_net(available_features).squeeze(1)
                        # Calculate average energy
                        view_energies[v] = energies.mean()
                else:
                    view_energies[v] = float('inf')  # If all samples are missing, set to infinity
            else:
                # If no missing information, calculate energy for all samples
                energies = self.energy_net(rs[v]).squeeze(1)
                view_energies[v] = energies.mean()

        return view_energies

    def _impute_features_and_update_mask(self, rs, missing_info):
        """
        Perform feature imputation at r-level using view energy differences and update missing mask
        For each view, find the reference view with minimal energy difference for imputation
        If no suitable reference view is found, no imputation is performed

        Args:
            rs: list of [batch_size, feature_dim] for each view
            missing_info: [batch_size, n_views], 1 means missing, 0 means available
        Returns:
            imputed_rs: list of imputed features
            updated_missing_info: updated missing mask
        """
        n_views = self.view
        device = missing_info.device

        # Check if any sample has all views missing
        all_missing = (missing_info.sum(dim=1) == n_views)
        if all_missing.any():
            raise ValueError("Some samples have all views missing, which is not supported.")

        # Stack features from all views, shape [batch_size, n_views, feature_dim]
        R = torch.stack(rs, dim=1)

        # Calculate average energy for each view
        view_energies = self._compute_view_energies(rs, missing_info)

        # Initialize updated missing mask (copy original mask)
        updated_missing = missing_info.clone()

        # Initialize result list
        imputed_rs = []
        for v in range(n_views):
            imputed_rs.append(rs[v].clone())

        # Create view availability mask: indicates if each view has at least one sample available
        view_available = torch.zeros(n_views, dtype=torch.bool, device=device)
        for v in range(n_views):
            view_available[v] = ((1 - missing_info[:, v]).sum() > 0)

        # Iterate through each view
        for target_view in range(n_views):
            # Check if target view has missing samples
            missing_count = missing_info[:, target_view].sum().item()
            if missing_count == 0:
                continue  # If no missing samples, skip this view

            # Calculate energy differences between current view and other views
            target_energy = view_energies[target_view]
            energy_diffs = torch.abs(target_energy - view_energies)

            # Set differences to infinity for self and unavailable views
            energy_diffs[target_view] = float('inf')  # Exclude self
            energy_diffs[~view_available] = float('inf')  # Exclude unavailable views

            # Find view with minimal energy difference
            best_view = torch.argmin(energy_diffs).item()

            # If minimal difference is infinity, no suitable reference view
            if energy_diffs[best_view] == float('inf'):
                continue

            # Find missing samples in target view
            missing_indices = torch.nonzero(missing_info[:, target_view]).squeeze(1)

            # Find available samples in reference view
            available_in_ref = (missing_info[:, best_view] == 0)

            # Only process samples missing in target view but available in reference view
            valid_indices = []
            for idx in missing_indices:
                if available_in_ref[idx]:
                    valid_indices.append(idx)

            if len(valid_indices) > 0:
                valid_indices = torch.tensor(valid_indices, device=device)

                # Get features from reference view
                ref_features = R[valid_indices, best_view]

                # Generate imputed features using imputation network
                imputed_features = self.imputation_nets[target_view](ref_features)

                # Update features
                imputed_rs[target_view][valid_indices] = imputed_features

                # Update missing mask
                updated_missing[valid_indices, target_view] = 0

        # Normalize all features
        for v in range(n_views):
            imputed_rs[v] = normalize(imputed_rs[v], dim=1)

        return imputed_rs, updated_missing

    def _process_views(self, views):
        """Process view data, extract features and reconstruct"""
        xrs, zs, rs = [], [], []

        for v in range(self.view):
            # Encode
            z = self.encoders[v](views[v])
            # Reconstruct
            xr = self.decoders[v](z)
            # Extract common features and normalize
            r = normalize(self.common_feature(z), dim=1)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        return xrs, zs, rs

    def _calculate_imputation_stats(self, missing_info, updated_missing_info):
        """
        Calculate imputation statistics for each view

        Args:
            missing_info: Original missing information mask [batch_size, n_views], 1 means missing, 0 means available
            updated_missing_info: Updated missing information mask [batch_size, n_views], 1 means missing, 0 means available

        Returns:
            imputation_stats: Dictionary containing imputation statistics for each view
                - total_missing: Total number of missing samples for each view
                - imputed: Number of successfully imputed samples for each view
                - ratio: Imputation ratio (imputed / total_missing)
        """
        n_views = self.view
        imputation_stats = {}

        for v in range(n_views):
            # Calculate original missing sample count
            total_missing = missing_info[:, v].sum().item()

            # Calculate imputed sample count (original missing=1, updated not missing=0)
            missing_original = (missing_info[:, v] == 1)
            not_missing_updated = (updated_missing_info[:, v] == 0)
            imputed_count = (missing_original & not_missing_updated).sum().item()

            # Calculate imputation ratio
            ratio = imputed_count / total_missing if total_missing > 0 else 0.0

            # Store statistics
            imputation_stats[f'view_{v}'] = {
                'total_missing': total_missing,
                'imputed': imputed_count,
                'ratio': ratio
            }

        # Add summary statistics for all views
        total_all_missing = missing_info.sum().item()
        total_all_imputed = ((missing_info == 1) & (updated_missing_info == 0)).sum().item()
        total_ratio = total_all_imputed / total_all_missing if total_all_missing > 0 else 0.0

        imputation_stats['overall'] = {
            'total_missing': total_all_missing,
            'imputed': total_all_imputed,
            'ratio': total_ratio
        }

        return imputation_stats

    def _compute_global_features_with_mask(self, rs, missing_mask):
        """
        Calculate global feature representation using updated missing mask and return attention weights

        Args:
            rs: list of [batch_size, feature_dim] for each view
            missing_mask: [batch_size, n_views], 1 means missing, 0 means available

        Returns:
            H_global: [batch_size, feature_dim] global feature representation
            attn_weights: [batch_size, n_views] attention weights
        """
        # Stack features from all views
        rs_stack = torch.stack(rs, dim=1)  # [batch_size, n_views, feature_dim]

        # Calculate attention weights using view_attention, considering missing mask
        attn_weights = self.view_attention(rs, missing_mask)  # [batch_size, n_views]

        # Expand attention weights to feature dimension for weighted sum
        attn_weights_expanded = attn_weights.unsqueeze(2)  # [batch_size, n_views, 1]

        # Weighted combination to get global features
        H_global = torch.sum(attn_weights_expanded * rs_stack, dim=1)  # [batch_size, feature_dim]

        return normalize(H_global, dim=1), attn_weights

    def forward(self, views, missing_info=None, mode='pretrain'):
        """
        Forward propagation process
        Args:
            views: List of view data
            missing_info: Missing information mask, 1 means missing, 0 means available
            mode: 'pretrain' - Pretraining mode, only reconstruction and basic feature extraction
                  'train' - Full training mode, including energy-based feature imputation and continuous missing mask update
        Returns:
            xrs: List of reconstructed data
            zs: List of features for each view
            imputed_rs: List of imputed features
            H: Global representation
            updated_missing_info: Updated missing mask (only returned in train mode)
            attn_weights: Attention weights (only returned in train mode)
            imputation_stats: Imputation statistics (only returned in train mode)
        """
        if mode == 'pretrain':
            xrs, zs, rs = self._process_views_with_mask(views, missing_info)
            # Calculate global representation and attention weights using non-imputed features
            H, _ = self._compute_global_features_with_mask(rs, missing_info)
            return xrs, zs, rs, H
        else:
            # Full training stage: use energy similarity to impute features for calculating H
            # First get original features
            xrs, zs, rs = self._process_views_with_mask(views,
                                                        missing_info) if missing_info is not None else self._process_views(
                views)

            if missing_info is not None:
                # Perform feature imputation based on energy similarity and update missing mask
                imputed_rs, updated_missing_info = self._impute_features_and_update_mask(rs, missing_info)

                # Calculate imputation statistics
                imputation_stats = self._calculate_imputation_stats(missing_info, updated_missing_info)

                # Calculate global representation and attention weights using updated missing mask and imputed features
                H, attn_weights = self._compute_global_features_with_mask(imputed_rs, updated_missing_info)

                return xrs, zs, imputed_rs, H, updated_missing_info, attn_weights, imputation_stats
            # else:
            #     # Case with no missing data
            #     imputed_rs = rs
            #
            #     # Create all-zero missing mask (indicating all data is available)
            #     fake_missing = torch.zeros((rs[0].size(0), self.view), device=rs[0].device)
            #
            #     # Create empty imputation statistics
            #     imputation_stats = {
            #         'overall': {'total_missing': 0, 'imputed': 0, 'ratio': 0.0}
            #     }
            #     for v in range(self.view):
            #         imputation_stats[f'view_{v}'] = {'total_missing': 0, 'imputed': 0, 'ratio': 0.0}
            #
            #     # Calculate global representation and attention weights
            #     H, attn_weights = self._compute_global_features_with_mask(imputed_rs, fake_missing)
            #
            #     return xrs, zs, imputed_rs, H, fake_missing, attn_weights, imputation_stats