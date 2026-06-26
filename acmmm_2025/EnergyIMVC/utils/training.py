import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import logging
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from utils.metrics import calculate_clustering_acc, calculate_purity
from losses.energy import compute_total_energy_loss
from losses.contrastive import EnergyEnhancedContrastiveLoss


class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reconstruction_loss = 0.0
        self.contrastive_loss = 0.0
        self.energy_loss = 0.0
        self.total_loss = 0.0
        self.count = 0

    def update(self, recon_loss=0.0, cont_loss=0.0, energy_loss=0.0, total_loss=0.0):
        self.reconstruction_loss += recon_loss
        self.contrastive_loss += cont_loss
        self.energy_loss += energy_loss
        self.total_loss += total_loss
        self.count += 1

    def get_averages(self):
        return {
            'recon': self.reconstruction_loss / self.count,
            'cont': self.contrastive_loss / self.count,
            'energy': self.energy_loss / self.count,
            'total': self.total_loss / self.count
        }


def pretrain(epoch, model, optimizer, data_loader, device):
    """Pretraining stage: only perform reconstruction"""
    model.train()
    tracker = LossTracker()
    criterion = nn.MSELoss(reduction='none')

    pbar = tqdm(data_loader,
                desc=f'Pretrain Epoch {epoch}',
                ncols=100,
                leave=True)

    for batch_idx, (xs, _, missing_info) in enumerate(pbar):
        xs = [x.to(device) for x in xs]
        missing_info = missing_info.to(device)

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs, missing_info, mode='pretrain')

        # Only consider non-missing data when calculating reconstruction loss
        total_recon_loss = 0.0
        total_elements = 0
        for v in range(model.view):
            available_mask = (1 - missing_info[:, v]).unsqueeze(1).float()
            recon_loss_per_element = criterion(xs[v], xrs[v])
            masked_loss = recon_loss_per_element * available_mask
            total_recon_loss += masked_loss.sum()
            total_elements += available_mask.sum()

        if total_elements > 0:
            avg_recon_loss = total_recon_loss / total_elements
        else:
            avg_recon_loss = torch.tensor(0.0, device=device)

        avg_recon_loss.backward()
        optimizer.step()

        tracker.update(
            recon_loss=avg_recon_loss.item(),
            total_loss=avg_recon_loss.item()
        )

        avg_losses = tracker.get_averages()
        pbar.set_postfix({
            'Recon': f'{avg_losses["recon"]:.4f}',
            'Total': f'{avg_losses["total"]:.4f}'
        })

    avg_losses = tracker.get_averages()
    logging.info(
        'Pretrain Epoch {} - Average Reconstruction Loss: {:.4f}'.format(
            epoch, avg_losses["recon"]
        )
    )


def contrastive_train(epoch, model, optimizer, data_loader, device,
                     temperature=1.0, energy_scale=0.1,
                     n_steps=50, step_size=0.1, noise_scale=0.05):
    """
    Contrastive learning training stage - Corrected contrastive loss calculation
     n_steps: Number of MCMC sampling steps
    step_size: MCMC step size
    noise_scale: Noise standard deviation
    """
    model.train()
    tracker = LossTracker()
    mse = nn.MSELoss(reduction='none')  # Use unreduced MSE to support sample-level weights

    # Initialize contrastive loss
    contrastive_loss = EnergyEnhancedContrastiveLoss(
        batch_size=data_loader.batch_size,
        temperature=temperature,
        device=device
    )

    # Only track overall imputation statistics
    total_missing = 0
    total_imputed = 0

    pbar = tqdm(data_loader,
                desc=f'Contrastive Epoch {epoch}',
                ncols=150,
                leave=True)

    for batch_idx, (xs, _, missing_info) in enumerate(pbar):
        xs = [x.to(device) for x in xs]
        missing_info = missing_info.to(device)

        optimizer.zero_grad()

        # Forward pass to get features, updated missing mask and imputation statistics
        outputs = model(xs, missing_info, mode='train')
        xrs, zs, imputed_rs, H, updated_missing_info, attn_weights, imputation_stats = outputs

        # Update overall imputation statistics
        total_missing += imputation_stats['overall']['total_missing']
        total_imputed += imputation_stats['overall']['imputed']

        # Calculate current batch imputation rate
        current_ratio = imputation_stats['overall']['ratio']

        # 1. Calculate total energy loss - pass updated_missing_info parameter and Energy network related parameters
        energy_loss = compute_total_energy_loss(
            model, xs, missing_info, H, imputed_rs, updated_missing_info,
            n_steps=n_steps,
            step_size=step_size,
            noise_scale=noise_scale
        )

        # 2. Calculate contrastive loss - Correction: no need to filter H, only filter unavailable r features
        cont_loss = 0.0
        for v in range(model.view):
            # Use updated missing mask to determine which samples have valid r features
            # Valid samples include: original non-missing samples and successfully imputed samples
            non_missing = (missing_info[:, v] == 0)  # Original non-missing samples
            missing_original = (missing_info[:, v] == 1)
            not_missing_updated = (updated_missing_info[:, v] == 0)
            imputed = missing_original & not_missing_updated

            # Combine the two types of valid samples
            valid_mask = non_missing | imputed

            # Skip views with no valid samples
            if valid_mask.sum() == 0:
                continue

            # Extract valid r features
            valid_r = imputed_rs[v][valid_mask]

            # Extract corresponding H features (H has no missing values)
            # Note: Here we only take H corresponding to valid_mask, because it needs to match valid_r samples
            valid_H = H[valid_mask]

            # If there are valid samples, calculate contrastive loss
            if len(valid_r) > 0:
                # Use attention weights to calculate contrastive loss for current view
                view_loss = contrastive_loss(
                    H=valid_H,
                    r=valid_r,
                    model=model,
                    energy_scale=energy_scale
                )
                cont_loss += view_loss

        # 3. Calculate reconstruction loss - use original missing_info
        recon_loss = 0.0
        for v in range(model.view):
            valid_mask = (missing_info[:, v] == 0).unsqueeze(1).float()
            view_recon_loss = mse(xs[v], xrs[v])
            masked_loss = view_recon_loss * valid_mask
            total_elements = valid_mask.sum()
            if total_elements > 0:
                view_loss = masked_loss.sum() / total_elements
            else:
                view_loss = torch.tensor(0.0, device=device)
            recon_loss += view_loss

        # 4. Combine all losses
        total_loss = cont_loss + recon_loss + energy_loss

        total_loss.backward()
        optimizer.step()

        # Update loss tracker
        tracker.update(
            recon_loss=recon_loss.item(),
            cont_loss=cont_loss.item(),
            energy_loss=energy_loss.item(),
            total_loss=total_loss.item()
        )

        # Update progress bar status - only show overall imputation rate
        avg_losses = tracker.get_averages()
        pbar.set_postfix({
            'Cont': f'{avg_losses["cont"]:.4f}',
            'Recon': f'{avg_losses["recon"]:.4f}',
            'Energy': f'{avg_losses["energy"]:.4f}',
            'Imputed': f'{current_ratio:.2%}'
        })

    # Calculate overall imputation rate
    overall_ratio = total_imputed / total_missing if total_missing > 0 else 0.0

    # Only output overall imputation rate
    logging.info(f'Epoch {epoch} Overall Imputation Rate: {overall_ratio:.2%} ({total_imputed}/{total_missing})')

    avg_losses = tracker.get_averages()
    logging.info(
        'Contrastive Epoch {} - Losses: Cont={:.4f}, Recon={:.4f}, Energy={:.4f}, Total={:.4f}'.format(
            epoch,
            avg_losses["cont"],
            avg_losses["recon"],
            avg_losses["energy"],
            avg_losses["total"]
        )
    )

    # Create simplified statistics dictionary, only including overall information
    simplified_stats = {
        'overall': {
            'total_missing': total_missing,
            'imputed': total_imputed,
            'ratio': overall_ratio
        }
    }

    return avg_losses, simplified_stats


def valid(model, device, data_loader, class_num, epoch):
    """Validation stage: perform clustering and evaluate performance"""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for xs, labels, missing_info in data_loader:
            xs = [x.to(device) for x in xs]
            missing_info = missing_info.to(device)

            # Call model's forward method to get outputs
            outputs = model(xs, missing_info, mode='train')

            # Check output format and extract H
            if len(outputs) >= 7:  # Full train mode output
                _, _, _, H, _, _, _ = outputs

            elif len(outputs) >= 5:  # Partial train mode output
                _, _, _, H, _ = outputs
            else:  # Pretrain mode output
                _, _, _, H = outputs

            all_features.append(H)
            all_labels.append(torch.as_tensor(labels).clone().detach())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logging.info('Clustering...')
    kmeans = KMeans(n_clusters=class_num, n_init=20)
    pred = kmeans.fit_predict(all_features.cpu().numpy())

    acc = calculate_clustering_acc(all_labels, pred)
    nmi = normalized_mutual_info_score(all_labels.cpu().numpy(), pred)
    pur = calculate_purity(all_labels, pred)

    logging.info('Epoch {} Metrics: ACC={:.4f}, NMI={:.4f}, PUR={:.4f}'.format(
        epoch, acc, nmi, pur))

    return acc, nmi, pur