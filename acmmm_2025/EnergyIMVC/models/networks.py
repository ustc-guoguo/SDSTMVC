import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


# -------------------------------
# ViewAttention
# -------------------------------
class ViewAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        """
        Args:
            feature_dim: Dimension of features for each view
            num_heads: Number of attention heads, feature_dim must be divisible by num_heads
            dropout: Dropout probability
        """
        super(ViewAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Generate Query and Key for each view separately, used for cross-view similarity calculation
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        # Scaling factor: reciprocal of square root of head_dim
        self.scale = self.head_dim ** -0.5

    def forward(self, view_features, missing_info=None):
        """
        Args:
            view_features: list of tensors, each tensor has shape [batch_size, feature_dim]
                           length is the number of views V
            missing_info: [batch_size, n_views], 1 means missing, 0 means available
        Returns:
            final_weights: [batch_size, V], attention weights for each view in each sample (normalized)
        """
        batch_size = view_features[0].size(0)
        n_views = len(view_features)

        # Stack to [batch_size, n_views, feature_dim]
        x = torch.stack(view_features, dim=1)

        # Calculate Query and Key
        Q = self.q_linear(x)  # [B, V, feature_dim]
        K = self.k_linear(x)  # [B, V, feature_dim]

        # Reshape to multi-head representation: [B, V, num_heads, head_dim] -> [B, num_heads, V, head_dim]
        Q = Q.view(batch_size, n_views, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, n_views, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores: dot product of Q and K^T for each head, shape [B, num_heads, V, V]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # If missing information is provided, create a mask matrix to mask the influence of missing views
        if missing_info is not None:
            # Calculate availability mask (1 means available, 0 means missing)
            avail_mask = 1 - missing_info  # [B, V]

            # Create view-to-view mask matrix [B, V, V]
            # Only allow attention calculation when both source and target views are available
            view_mask = avail_mask.unsqueeze(2) * avail_mask.unsqueeze(1)  # [B, V, V]

            # Expand to multi-head format [B, num_heads, V, V]
            view_mask = view_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Apply mask, set attention scores between unavailable view pairs to extremely small value
            attn_scores = attn_scores.masked_fill(view_mask == 0, -1e9)

        # Apply softmax on the last dimension for each head to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Average multi-head attention weights to get [B, V, V] cross-view attention matrix
        attn_weights_avg = attn_weights.mean(dim=1)

        # Use diagonal elements (representing each view's attention to itself) as initial scores
        self_scores = torch.diagonal(attn_weights_avg, dim1=1, dim2=2)  # [B, V]

        # If missing information is provided, ensure missing views get zero scores
        if missing_info is not None:
            # Apply view availability mask to ensure missing views get score 0
            self_scores = self_scores * avail_mask

        # Handle all-zero rows (case where all views are missing)
        zero_rows = (self_scores.sum(dim=1) == 0).unsqueeze(1)  # [B, 1]
        uniform_weights = torch.ones_like(self_scores) / n_views

        # Apply softmax to non-zero rows
        softmax_weights = F.softmax(self_scores, dim=-1)

        # Combine results: use uniform weights for all-zero rows, otherwise use softmax weights
        final_weights = torch.where(zero_rows, uniform_weights, softmax_weights)
        # print(final_weights)

        return final_weights


class ImputationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ImputationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Initialize weights and biases
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for weights
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

                # Initialize biases to small positive values
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Use 0.1 as initial value

    def forward(self, x):
        """
        Args:
            x: Concatenated features [batch_size, input_dim*2]
        """
        # Directly generate imputed features through the network
        return self.net(x)


class EnergyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(EnergyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # Apply Kaiming initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization (normal distribution)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Use smaller weight initialization for the last layer
                if m == self.net[-2]:  # The last linear layer is the second-to-last module
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Use small positive bias

    def forward(self, x):
        return self.net(x)