import torch
import torch.nn.functional as F


def compute_imputation_energy_loss(model, views, missing_info, imputed_rs=None, updated_missing_info=None, epoch=None):
    """Calculate view-level energy loss, excluding samples that haven't been imputed
    Args:
        model: MultiViewClusteringModel instance
        views: List of original view data
        missing_info: Original missing information mask (1 means missing, 0 means available)
        imputed_rs: List of imputed r-layer features
        updated_missing_info: Updated missing information mask (1 means missing, 0 means available)
        epoch: Current training epoch, used for dynamically adjusting weights
    Returns:
        imp_loss: Energy loss value
    """
    if missing_info is None or imputed_rs is None:
        return torch.tensor(0.0, device=views[0].device)

    # If updated_missing_info is not provided, use the original missing_info
    if updated_missing_info is None:
        updated_missing_info = missing_info.clone()

    # Dynamically adjust weight coefficient
    energy_weight = 1.0
    if epoch is not None:
        energy_weight = min(1.0, epoch / 50.0)  # Gradually increase weight for first 50 epochs

    intra_view_loss = 0.0
    device = views[0].device

    for v in range(model.view):
        # Calculate original r features
        z = model.encoders[v](views[v])
        r_true = F.normalize(model.common_feature(z), dim=1)

        # Get imputed r features
        r_imputed = imputed_rs[v]

        # Get masks for three types of samples
        # 1. Non-missing samples (original_missing = 0)
        non_missing_mask = (missing_info[:, v] == 0)

        # 2. Imputed samples (original missing=1, updated not missing=0)
        missing_original = (missing_info[:, v] == 1)
        not_missing_updated = (updated_missing_info[:, v] == 0)
        imputed_mask = missing_original & not_missing_updated

        # Check if there are non-missing samples and imputed samples
        has_non_missing = non_missing_mask.sum() > 0
        has_imputed = imputed_mask.sum() > 0

        if has_non_missing and has_imputed:
            # Calculate original feature energy only for non-missing samples
            r_true_observed = r_true[non_missing_mask]

            # Calculate imputed feature energy only for imputed samples
            r_imputed_valid = r_imputed[imputed_mask]

            # Calculate energy
            E_true_observed = model.energy_net(r_true_observed)
            E_imputed_valid = model.energy_net(r_imputed_valid)

            # Calculate expected energy difference
            E_true_observed_mean = torch.mean(E_true_observed)
            E_imputed_valid_mean = torch.mean(E_imputed_valid)

            # Squared difference plus regularization term
            energy_similarity = (E_true_observed_mean - E_imputed_valid_mean) ** 2
            energy_constraint = 0.01 * E_imputed_valid_mean ** 2

            # View loss
            view_loss = energy_similarity + energy_constraint
            intra_view_loss += view_loss
        elif has_non_missing:
            # Only non-missing samples, no imputation performed - skip this view
            continue
        else:
            # No non-missing samples - skip this view
            continue

    # Calculate cross-view energy consistency loss - only consider valid views
    cross_view_loss = 0.0
    view_count = 0

    # Find valid views (at least one non-missing or imputed sample)
    valid_views = []
    view_energies = []

    for v in range(model.view):
        # Calculate number of non-missing samples
        non_missing_count = (missing_info[:, v] == 0).sum().item()

        # Calculate number of imputed samples (original missing=1, updated not missing=0)
        missing_original = (missing_info[:, v] == 1)
        not_missing_updated = (updated_missing_info[:, v] == 0)
        imputed_count = (missing_original & not_missing_updated).sum().item()

        if non_missing_count > 0 or imputed_count > 0:
            valid_views.append(v)

            # Create mask for valid samples (non-missing or imputed)
            non_missing_mask = (missing_info[:, v] == 0)
            imputed_mask = missing_original & not_missing_updated
            valid_mask = non_missing_mask | imputed_mask

            # Only calculate energy for valid samples
            valid_features = imputed_rs[v][valid_mask]
            if len(valid_features) > 0:
                view_energy = torch.mean(model.energy_net(valid_features))
                view_energies.append(view_energy)
                view_count += 1

    # If there are at least two valid views, calculate energy difference between them
    if view_count >= 2:
        for i in range(len(valid_views)):
            for j in range(i + 1, len(valid_views)):
                energy_diff = (view_energies[i] - view_energies[j]) ** 2
                cross_view_loss += energy_diff

    # Combine losses, apply dynamic weight
    # If there are no valid views, return zero loss
    if view_count == 0:
        return torch.tensor(0.0, device=device)

    imp_loss = energy_weight * (intra_view_loss + cross_view_loss)

    return imp_loss


def sample_from_energy_net(model, x, n_steps=50, step_size=0.1, noise_scale=0.005):
    """Sample from energy network using Langevin dynamics
    Args:
        model: MultiViewClusteringModel instance
        x: Initial features [batch_size, feature_dim]
        n_steps: Number of MCMC sampling steps
        step_size: MCMC step size
        noise_scale: Noise standard deviation
    Returns:
        Sampled features
    """
    x_k = x.clone().detach()

    for _ in range(n_steps):
        x_k.requires_grad_(True)
        energy = model.energy_net(x_k)
        grad_x = torch.autograd.grad(energy.sum(), x_k)[0]

        # Langevin dynamics
        noise = torch.randn_like(x_k) * noise_scale
        x_k = x_k - step_size * grad_x + noise
        x_k = x_k.detach()

        # Keep on unit sphere
        x_k = F.normalize(x_k, dim=1)

    return x_k


def compute_global_energy_loss(model, H, updated_missing_info=None, n_steps=50, step_size=0.1, noise_scale=0.05):
    """Calculate global feature energy loss, excluding completely missing samples
    Args:
        model: MultiViewClusteringModel instance
        H: Global features
        updated_missing_info: Updated missing information mask (marking which samples have been imputed)
        n_steps: Number of MCMC sampling steps
        step_size: MCMC step size
        noise_scale: Noise standard deviation
    """
    # Check if there are completely missing samples (all views are missing)
    # if updated_missing_info is not None:
    #     # Find samples with at least one non-missing view
    #     valid_samples = (updated_missing_info.sum(dim=1) < model.view).bool()
    #
    #     # If all are invalid, return zero loss
    #     if valid_samples.sum() == 0:
    #         return torch.tensor(0.0, device=H.device)

    #     # Only calculate energy for valid samples
    #     H_valid = H[valid_samples]
    # else:
    #     # If no mask is provided, assume all samples are valid

    # Here obviously our H cannot have missing cases, so no need to consider valid situations.
    H_valid = H

    # Calculate energy of data samples
    E_data = model.energy_net(H_valid)

    # Use MCMC to generate model samples
    H_model = sample_from_energy_net(model, H_valid,
                                     n_steps=n_steps,
                                     step_size=step_size,
                                     noise_scale=noise_scale)
    E_model = model.energy_net(H_model)

    # Calculate contrastive divergence loss
    cd_loss = torch.mean(F.softplus(E_data - E_model + model.margin))

    return cd_loss


def compute_total_energy_loss(model, views, missing_info, H, imputed_rs=None, updated_missing_info=None,
                              n_steps=50, step_size=0.1, noise_scale=0.05):
    """Calculate total energy loss, excluding samples that haven't been imputed
    Args:
        model: MultiViewClusteringModel instance
        views: List of original view data
        missing_info: Original missing information mask
        H: Global features
        imputed_rs: List of already imputed r-layer features
        updated_missing_info: Updated missing information mask
        step_size: MCMC step size
        noise_scale: MCMC noise scale
    Returns:
        total_energy_loss: A 0-dimensional torch.Tensor representing the total energy loss
    """
    # Use explicit keyword arguments to avoid parameter order issues
    imp_loss = compute_imputation_energy_loss(
        model=model,
        views=views,
        missing_info=missing_info,
        imputed_rs=imputed_rs,
        updated_missing_info=updated_missing_info
    )

    # Calculate global energy loss
    cd_loss = compute_global_energy_loss(model, H, updated_missing_info, n_steps=n_steps, step_size=step_size,
                                         noise_scale=noise_scale)
    # Combine losses
    total_energy_loss = model.imputation_loss_weight * imp_loss + model.cd_loss_weight * cd_loss

    return total_energy_loss