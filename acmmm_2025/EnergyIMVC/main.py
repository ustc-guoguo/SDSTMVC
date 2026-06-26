import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "7"
import torch
import random
import logging
import numpy as np
import argparse
from torch.optim import Adam
from models.clustering import MultiViewClusteringModel
from utils.training import pretrain, contrastive_train, valid
from dataprocessing import MultiviewData, get_multiview_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset_seed(dataset):
    """Get dataset-specific random seed."""
    seeds = {
        "MNIST-USPS": 10,
        "BDGP": 30,
        "Fashion": 10,
        "MSRCv1": 42,
        "hand": 42
    }
    return seeds.get(dataset, 42)


def save_model(model, dataset):
    """Save model to disk."""
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    torch.save(
        model.state_dict(),
        os.path.join('./checkpoints', f'{dataset}.pth')
    )
    logging.info(f"Checkpoint saved for {dataset}.")


def main():
    parser = argparse.ArgumentParser(
        description='Train or Evaluate Multi-view Clustering Model with Joint Energy-based Imputation')

    # Dataset related
    parser.add_argument('--dataset', default='BDGP', type=str,
                        help='Name of the dataset')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for training/evaluation')
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='Path to dataset directory')

    # Training related
    parser.add_argument('--learning_rate', default=0.003, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--pre_epochs', default=200, type=int,
                        help='Number of pre-training epochs')
    parser.add_argument('--con_epochs', default=100, type=int,
                        help='Number of contrastive training epochs')

    # Model architecture related
    parser.add_argument('--feature_dim', default=64, type=int,
                        help='Dimension of feature space')
    parser.add_argument('--high_feature_dim', default=20, type=int,
                        help='Dimension of high-level feature space')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Hidden dimension for networks')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--num_heads', default=4, type=int,
                        help='num of heads for multi head attention mechanism')

    # Loss function weights
    parser.add_argument('--imputation_loss_weight', default=0.1, type=float,
                        help='Weight for imputation loss')
    parser.add_argument('--cd_loss_weight', default=0.1, type=float,
                        help='Weight for contrastive divergence loss')

    # Energy network related
    parser.add_argument('--margin', default=1, type=float,
                        help='Margin for energy-based loss')
    parser.add_argument('--n_steps', default=50, type=int,
                        help='Number of steps for MCMC sampling')
    parser.add_argument('--step_size', default=0.1, type=float,
                        help='Step size for MCMC sampling')
    parser.add_argument('--noise_scale', default=0.05, type=float,
                        help='Noise scale for MCMC sampling')

    # Contrastive learning related
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature parameter for contrastive loss')
    parser.add_argument('--energy_scale', default=0.1, type=float,
                        help='Energy scale for Energy-Enhanced Contrastive loss')

    # New parameter: whether to directly load checkpoint for evaluation
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='If set, directly load checkpoint from ./checkpoints and evaluate.')

    args = parser.parse_args()

    # Set random seed
    seed = get_dataset_seed(args.dataset)
    setup_seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: {}'.format(device))

    # 1. 数据
    mv_data = MultiviewData(args.dataset, device, path=args.data_path)
    data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)
    logging.info('Dataset loaded: {} views, {} samples, {} clusters'.format(
        num_views, num_samples, num_clusters))

    # Get dimensions for each view
    view_dims = [mv_data.data_views[i].shape[1] for i in range(num_views)]

    # 2. 模型
    model = MultiViewClusteringModel(
        view_dims=view_dims,
        feature_dim=args.feature_dim,
        high_feature_dim=args.high_feature_dim,
        margin=args.margin,
        imputation_loss_weight=args.imputation_loss_weight,
        cd_loss_weight=args.cd_loss_weight,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    ).to(device)

    # If load_checkpoint is set, load and evaluate without training
    if args.load_checkpoint:
        checkpoint_path = os.path.join('./checkpoints', f'{args.dataset}.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            logging.info("Checkpoint loaded. Evaluating model...")
            # Modified here: receiving 4 return values but only using the first 3
            acc, nmi, pur, _ = valid(model, device, data_loader, num_clusters, epoch=0)
            logging.info("Evaluation Results: ACC = {:.4f}, NMI = {:.4f}, PUR = {:.4f}".format(acc, nmi, pur))
            return
        else:
            logging.info(f"No checkpoint found at {checkpoint_path}. Exiting.")
            return

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_acc = 0
    best_nmi = 0
    best_pur = 0
    epoch = 1

    # Pretraining stage
    while epoch <= args.pre_epochs:
        pretrain(epoch, model, optimizer, data_loader, device)
        epoch += 1

    # Contrastive learning stage
    while epoch <= args.pre_epochs + args.con_epochs:
        # Call contrastive_train
        avg_losses, _ = contrastive_train(
            epoch, model, optimizer, data_loader, device,
            temperature=args.temperature,
            energy_scale=args.energy_scale,
            n_steps=args.n_steps,
            step_size=args.step_size,
            noise_scale=args.noise_scale
        )

        # Validation
        acc, nmi, pur = valid(model, device, data_loader, num_clusters, epoch)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_pur = pur
            save_model(model, args.dataset)

        epoch += 1

    logging.info('Best Results: ACC = {:.4f}, NMI = {:.4f}, PUR = {:.4f}'.format(
        best_acc, best_nmi, best_pur))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    main()