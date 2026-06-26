import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "7"
import torch
import logging
import argparse
import numpy as np
import optuna
from optuna.trial import Trial
from torch.optim import Adam

# Import project components
from models.clustering import MultiViewClusteringModel
from utils.training import pretrain, contrastive_train, valid
from dataprocessing import MultiviewData, get_multiview_data


def setup_logging():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("hyperparameter_tuning.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("OptunaTuning")


def setup_seed(seed):
    """Set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset_seed(dataset):
    """Get dataset-specific random seed"""
    seeds = {
        "MNIST-USPS": 42,
        "BDGP": 42,
        "Fashion": 42,
        "hand": 42
    }
    return seeds.get(dataset, 42)


def objective(trial: Trial, args, dataset_name):
    """Optuna optimization objective function"""
    logger = logging.getLogger("OptunaTuning")
    logger.info(f"Starting trial {trial.number} for dataset {dataset_name}")

    # First decide num_heads, so we can define high_feature_dim based on it
    num_heads = trial.suggest_int('num_heads', 1, 8)

    # Ensure high_feature_dim is divisible by num_heads
    # Define a range of possible dimensions, all divisible by num_heads
    possible_dims = [dim for dim in range(num_heads, 128, num_heads)]

    # If possible_dims is empty (extreme case), use default value
    if not possible_dims:
        high_feature_dim = num_heads * 4  # Default value, ensures divisibility by num_heads
    else:
        # Choose one dimension from the possible dimensions
        high_feature_dim_index = trial.suggest_int('high_feature_dim_index', 0, len(possible_dims) - 1)
        high_feature_dim = possible_dims[high_feature_dim_index]

    # Define other hyperparameter search spaces
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'feature_dim': trial.suggest_int('feature_dim', 32, 128),
        'high_feature_dim': high_feature_dim,  # Use the calculated high_feature_dim
        'num_heads': num_heads,  # Use the already decided num_heads
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'imputation_loss_weight': trial.suggest_float('imputation_loss_weight', 0.01, 0.5),
        'cd_loss_weight': trial.suggest_float('cd_loss_weight', 0.01, 0.5),
        'temperature': trial.suggest_float('temperature', 0.1, 2.0),
        'energy_scale': trial.suggest_float('energy_scale', 0.01, 1.0),
        'n_steps': trial.suggest_int('n_steps', 10, 100),
        'step_size': trial.suggest_float('step_size', 0.01, 0.5),
        'noise_scale': trial.suggest_float('noise_scale', 0.001, 0.1, log=True),
        'margin': trial.suggest_float('margin', 0.1, 3.0)
    }

    # Log current trial parameters
    logger.info(f"Trial parameters for {dataset_name}: {params}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    seed = get_dataset_seed(dataset_name)
    setup_seed(seed)

    # Load dataset
    mv_data = MultiviewData(dataset_name, device, path=args.data_path)
    data_loader, num_views, num_samples, num_clusters = get_multiview_data(
        mv_data, params['batch_size'])

    # Get dimensions for each view
    view_dims = [mv_data.data_views[i].shape[1] for i in range(num_views)]

    # Initialize model
    model = MultiViewClusteringModel(
        view_dims=view_dims,
        feature_dim=params['feature_dim'],
        high_feature_dim=params['high_feature_dim'],
        margin=params['margin'],
        imputation_loss_weight=params['imputation_loss_weight'],
        cd_loss_weight=params['cd_loss_weight'],
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout'],
        num_heads=params['num_heads']
    ).to(device)

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=args.weight_decay
    )

    # Training loop
    best_acc = 0
    epoch = 1

    # Pretraining phase (reduce epochs to speed up parameter tuning)
    pre_epochs = min(args.pre_epochs, 50)  # Reduce epochs
    while epoch <= pre_epochs:
        pretrain(epoch, model, optimizer, data_loader, device)
        epoch += 1

        # Report intermediate progress every 10 epochs
        if epoch % 10 == 0:
            intermediate_acc, _, _ = valid(model, device, data_loader, num_clusters, epoch)
            trial.report(intermediate_acc, epoch)

            # Early terminate poorly performing trials
            if trial.should_prune():
                logger.info(f"Trial {trial.number} for {dataset_name} pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

    # Contrastive learning phase (reduce epochs to speed up parameter tuning)
    con_epochs = min(args.con_epochs, 50)  # Reduce epochs
    while epoch <= pre_epochs + con_epochs:
        # Update contrastive_train call, passing all necessary parameters
        avg_losses, _ = contrastive_train(
            epoch, model, optimizer, data_loader, device,
            temperature=params['temperature'],
            energy_scale=params['energy_scale'],
            n_steps=params['n_steps'],
            step_size=params['step_size'],
            noise_scale=params['noise_scale']
        )

        # Validation
        acc, nmi, pur = valid(model, device, data_loader, num_clusters, epoch)

        if acc > best_acc:
            best_acc = acc

        # Report progress
        trial.report(acc, epoch)

        # Early terminate poorly performing trials
        if trial.should_prune():
            logger.info(f"Trial {trial.number} for {dataset_name} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        epoch += 1

    logger.info(f"Trial {trial.number} for {dataset_name} finished with best accuracy: {best_acc}")
    return best_acc


def save_best_params(study, dataset_name, filename=None):
    """Save best parameters to file"""
    if filename is None:
        filename = f"{dataset_name}_best_params.txt"

    with open(filename, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best accuracy: {study.best_value}\n")
        f.write("Best parameters:\n")

        for key, value in study.best_params.items():
            f.write(f"    --{key}={value}\n")

        # Add command line format parameters
        f.write("\nCommand format:\n")
        cmd = f"python main.py --dataset={dataset_name} "
        for key, value in study.best_params.items():
            if key != 'batch_size' and key != 'high_feature_dim_index':  # Exclude batch_size and index parameters
                cmd += f"--{key}={value} "
        f.write(cmd + "\n")


def run_optimization_for_dataset(args, dataset_name):
    """Run optimization for a single dataset"""
    logger = logging.getLogger("OptunaTuning")
    logger.info(f"Starting hyperparameter tuning for {dataset_name} dataset")

    # Create or load storage
    if args.storage:
        logger.info(f"Using storage at {args.storage}")
        study_name = f"{dataset_name}_optimization"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            load_if_exists=True,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
    else:
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )

    # Run optimization
    n_trials = args.n_trials_per_dataset
    logger.info(f"Starting optimization for {dataset_name} with {n_trials} trials")
    study.optimize(lambda trial: objective(trial, args, dataset_name), n_trials=n_trials)

    # Print and save results
    logger.info(f"Optimization for {dataset_name} finished")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best accuracy: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")

    # Save best parameters
    save_best_params(study, dataset_name)

    # Plot optimization results
    try:
        import matplotlib.pyplot as plt

        # Importance plot
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(f"{dataset_name}_param_importances.png")
        plt.close()

        # History plot
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(f"{dataset_name}_optimization_history.png")
        plt.close()

        # Parameter relationship plots
        important_params = ['learning_rate', 'feature_dim', 'imputation_loss_weight',
                            'cd_loss_weight', 'margin', 'energy_scale', 'num_heads']
        for param in important_params:
            try:
                fig = optuna.visualization.matplotlib.plot_slice(study, params=[param])
                plt.savefig(f"{dataset_name}_{param}_slice.png")
                plt.close()
            except:
                logger.warning(f"Could not generate slice plot for {param}")

    except Exception as e:
        logger.error(f"Error generating plots for {dataset_name}: {e}")

    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning for Multi-view Clustering Model on Multiple Datasets')

    # Fixed parameters
    parser.add_argument('--datasets', nargs='+', default=['MNIST-USPS', 'BDGP', 'Fashion', 'hand'],
                        help='Names of datasets to optimize')
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='Path to dataset directory')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--pre_epochs', default=200, type=int,
                        help='Maximum number of pre-training epochs for tuning')
    parser.add_argument('--con_epochs', default=200, type=int,
                        help='Maximum number of contrastive training epochs for tuning')
    parser.add_argument('--n_trials_per_dataset', default=100, type=int,
                        help='Number of trials for optimization per dataset')
    parser.add_argument('--storage', default=None, type=str,
                        help='Storage URL for Optuna database')

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting hyperparameter tuning for datasets: {args.datasets}")

    # Store best parameters for all datasets
    all_best_params = {}

    # Run optimization for each dataset
    for dataset in args.datasets:
        logger.info(f"==== Starting optimization for {dataset} ====")
        best_params, best_acc = run_optimization_for_dataset(args, dataset)
        all_best_params[dataset] = {
            'params': best_params,
            'accuracy': best_acc
        }
        logger.info(f"==== Completed optimization for {dataset} ====")

    # Save best parameters for all datasets to a file
    with open("all_datasets_best_params.txt", "w") as f:
        f.write("Best parameters for all datasets:\n\n")
        for dataset, results in all_best_params.items():
            f.write(f"==== {dataset} (Accuracy: {results['accuracy']:.4f}) ====\n")
            for key, value in results['params'].items():
                if key != 'high_feature_dim_index':  # Exclude index parameter
                    f.write(f"    --{key}={value}\n")
            f.write("\n")

            # Add command line format
            cmd = f"python main.py --dataset={dataset} "
            for key, value in results['params'].items():
                if key != 'batch_size' and key != 'high_feature_dim_index':  # Exclude batch_size and index parameters
                    cmd += f"--{key}={value} "
            f.write(f"Command: {cmd}\n\n")

    logger.info("Completed optimization for all datasets. Results saved to all_datasets_best_params.txt")


if __name__ == "__main__":
    main()