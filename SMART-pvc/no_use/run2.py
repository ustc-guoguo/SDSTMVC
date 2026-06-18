import argparse
import collections
import random
from datasets import *
from configure import set_default_config, Config, set_config
import warnings

warnings.simplefilter("ignore")


def main(config):
    from model import OTGM, Model, Model_Shared, Model_Shared0
    from train import pretrain, train
    from utils import cal_std, get_logger, set_random_seed

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger = get_logger(root=f"./training_logs/{config['dataset']}",
                        filename=config['log_name'])
    # Load data
    data_root = "D:/LPeng/work_space/py_space/pytorch110/Data/MVDATA"
    X_list, Y_list = load_data(config, root=data_root)
    Y_list = Y_list[0]
    # logger.info(f"X1: {X_list[0].shape}")           # X1: (2000, 240)
    # logger.info(f"X2: {X_list[1].shape}")           # X2: (2000, 216)
    # logger.info(f"Y: {Y_list}")                     # Y: [array([0, 0, 0, ..., 9, 9, 9])]
    x1_train = torch.from_numpy(X_list[0]).float().to(device)
    x2_train = torch.from_numpy(X_list[1]).float().to(device)
    config['n_samples'] = x1_train.shape[0]
    config['n_classes'] = len(np.unique(Y_list))
    config['attribute1'] = X_list[0].shape[1]
    config['attribute2'] = X_list[1].shape[1]
    config['in_dim'][0], config['in_dim'][1] = x1_train.shape[1], x2_train.shape[1]
    flag = get_aligned(x1_train.shape[0], config['aligned_ratio'])

    logger.info('Config:')
    for (key, value) in config.items():
        logger.info(f'- {key:>15} : {value}')

    best_result = {
        'acc': [], 'nmi': [], 'ari': [], 'f1': [],
        'acc_1': [], 'nmi_1': [], 'ari_1': [], 'f1_1': [],
        'acc_2': [], 'nmi_2': [], 'ari_2': [], 'f1_2': []
    }

    # Seeds for each time of training
    if config['train']:
        # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # seeds = range(config['seed'], config['seed']+91, 10)
        seeds = [5, 15, 25, 35, 45]
    else:
        seeds = [5]
    for seed in seeds:
        # Set random seeds
        set_random_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed + 1)
        # torch.manual_seed(seed + 2)
        # torch.cuda.manual_seed(seed + 3)
        # torch.backends.cudnn.deterministic = True
        logger.info(f'========================= SEED {seed} =========================')

        # Build model
        # model = OTGM(config)
        # model = Model(in_dim=config['in_dim'],
        #              hidden_dim=config['hidden_dim'],
        #              emb_dim=config['emb_dim'],
        #              p_dim=config['p_dim'],
        #              n_layers_e=config['n_layers_e'],
        #              n_layers_d=config['n_layers_d'],
        #              activation=config['activation'],
        #              batchnorm=config['batchnorm'],
        #              dropout=config['dropout'],
        #              device=device)
        model = Model_Shared(in_dim=config['in_dim'],
                             hidden_dim=config['hidden_dim'],
                             emb_dim=config['emb_dim'],
                             p_dim=config['p_dim'],
                             n_layers_e=config['n_layers_e'],
                             n_layers_d=config['n_layers_d'],
                             activation=config['activation'],
                             batchnorm=config['batchnorm'],
                             dropout=config['dropout'],
                             device=device)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=float(config['lr']),
                                     weight_decay=float(config['weight_decay']))
        if seed == seeds[0]:
            # logger.info(model)
            logger.info(f"Weight Shared? {model.encoder1[-1].weight is model.encoder2[-1].weight}")     # Weight Shared? True
            # logger.info(optimizer)
        # # pretrain
        # pretrain_dir = 'pretrain/%s/rate_%s' % (config['dataset'], config['training']['aligned_ratio'])
        # if not os.path.exists(pretrain_dir):
        #     os.mkdir(pretrain_dir)
        # pretrain_name = '%s_alpha_%s_l1_%s_l2_%s_%s.pkl' % (config['dataset'], config['training']['alpha'], config['training']['lambda1'],
        #     config['training']['lambda2'], config['training']['pre_name'])
        # pretrain_path = os.path.join(pretrain_dir, pretrain_name)
        # if not os.path.exists(pretrain_path + 'a'):
        #     pretrain(model.network, optimizer, config, x1_train, x2_train, flag, Y_list, logger, pretrain_path=pretrain_path,
        #              device=device)
        # else:
        #     model.network.load_state_dict(torch.load(pretrain_path))
        # # shuffle
        # model.network.evaluation(logger, x1_train, x2_train, Y_list)
        x1_train, x2_train, P_index, index_mis_aligned, P_gt = get_mis_aligned(x1_train, x2_train, flag, device)
        # logger.info(f"flag: {flag.shape}\n{flag}")                  # flag: (2000,) [ True False  True ... ]
        # logger.info(f"P_index: {P_index.shape}\n{P_index}")         # P_index: (2000,)
        # logger.info(f"index_mis_aligned: {index_mis_aligned.shape}\n{index_mis_aligned}")       # index_mis_aligned: (1000,)
        # train
        best, best_1, best_2 = train(model, optimizer, config, x1_train, x2_train, flag, Y_list, index_mis_aligned, P_index, logger, device)

        best_result['acc'].append(best[0])
        best_result['nmi'].append(best[1])
        best_result['ari'].append(best[2])
        best_result['f1'].append(best[3])
        best_result['acc_1'].append(best_1[0])
        best_result['nmi_1'].append(best_1[1])
        best_result['ari_1'].append(best_1[2])
        best_result['f1_1'].append(best_1[3])
        best_result['acc_2'].append(best_2[0])
        best_result['nmi_2'].append(best_2[1])
        best_result['ari_2'].append(best_2[2])
        best_result['f1_2'].append(best_2[3])

    logger.info(f"Dataset: {config['dataset']}, alr: {config['aligned_ratio']}")
    logger.info(f"lr: {config['lr']}, ne: {config['n_layers_e']}, nd: {config['n_layers_d']}")
    logger.info(f"L1: {config['lambda1']}, L2: {config['lambda2']}, t1: {config['threshold1']}, t2: {config['threshold2']}, tau: {config['tau']}")
    logger.info(f"hid: {config['hidden_dim']}, emb: {config['emb_dim']}, prj: {config['p_dim']}, fu: {config['fusion_mode']}/{config['fusion_ratio']}")
    # logger.info(f"hid {args.hid_dim}, fm {config['fusion_mode']}, fr {config['fusion_ratio']}")
    logger.info(f"- ACC: {np.mean(best_result['acc']) * 100:.2f} ± {np.std(best_result['acc']) * 100:.2f}")
    logger.info(f"- NMI: {np.mean(best_result['nmi']) * 100:.2f} ± {np.std(best_result['nmi']) * 100:.2f}")
    logger.info(f"- ARI: {np.mean(best_result['ari']) * 100:.2f} ± {np.std(best_result['ari']) * 100:.2f}")
    logger.info(f"- F1 : {np.mean(best_result['f1']) * 100:.2f} ± {np.std(best_result['f1']) * 100:.2f}")
    logger.info(f"V1: ")
    logger.info(f"- ACC: {np.mean(best_result['acc_1']) * 100:.2f} ± {np.std(best_result['acc_1']) * 100:.2f}")
    logger.info(f"- NMI: {np.mean(best_result['nmi_1']) * 100:.2f} ± {np.std(best_result['nmi_1']) * 100:.2f}")
    logger.info(f"- ARI: {np.mean(best_result['ari_1']) * 100:.2f} ± {np.std(best_result['ari_1']) * 100:.2f}")
    logger.info(f"- F1 : {np.mean(best_result['f1_1']) * 100:.2f} ± {np.std(best_result['f1_1']) * 100:.2f}")
    logger.info(f"V2: ")
    logger.info(f"- ACC: {np.mean(best_result['acc_2']) * 100:.2f} ± {np.std(best_result['acc_2']) * 100:.2f}")
    logger.info(f"- NMI: {np.mean(best_result['nmi_2']) * 100:.2f} ± {np.std(best_result['nmi_2']) * 100:.2f}")
    logger.info(f"- ARI: {np.mean(best_result['ari_2']) * 100:.2f} ± {np.std(best_result['ari_2']) * 100:.2f}")
    logger.info(f"- F1 : {np.mean(best_result['f1_2']) * 100:.2f} ± {np.std(best_result['f1_2']) * 100:.2f}")
    logger.info(f"========================= Training Over =======================")


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', help='name of dataset.',
                        choices=['cora', 'citeseer', 'pubmed', 'cornell', 'texas', 'wisconsin', 'film', 'squirrel',
                                 'chameleon'])
    parser.add_argument('--datasetid', type=int, default=0, help='Dataset number in datasets dictionary.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--hidden_dim', type=int, default=1500, help='Dimensions of the hidden layers')
    parser.add_argument('--emb_dim', type=int, default=50, help='Dimension of the learned embeddings')
    parser.add_argument('--p_dim', type=int, default=50, help='Output dim of the projectors')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay rate.')
    parser.add_argument('--lambda1', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--threshold', type=float, default=0.5, help='.')
    parser.add_argument('--k', type=float, default=0.1, help='.')
    parser.add_argument('--tau', type=float, default=0.1, help='Similarity distribution temperature factor.')
    parser.add_argument('--fusion_ratio', type=float, default=0.8, help='fusion ratio.')
    parser.add_argument('--deviceid', type=str, default='0', help='id of training device.')

    parser.add_argument('--testid', type=int, default=0, help='just for test.')

    args = parser.parse_args()

    datasets = {
        0: "HandWritten", 1: "Scene-15", 2: "BDGP",
        3: "Caltech101-7", 4: "Caltech101-20", 5: "Reuters_dim10",
    }
    # Set physical GPU
    args.deviceid = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.deviceid)

    args.datasetid = 0
    args.dataset = datasets[args.datasetid]

    # config = set_default_config(args.dataset)
    config = set_config(args.dataset)
    config['train'] = True

    config['lr'] = 0.0001
    config['aligned_ratio'] = 0.0
    config['lambda0'] = 1.0
    config['lambda1'] = 0
    config['lambda2'] = 10
    config['threshold1'] = 0.2
    config['threshold2'] = 0.2
    # config['alpha1'] = args.alpha
    # config['alpha2'] = args.alpha
    config['tau'] = 1.0
    config['fusion_mode'] = 2
    config['fusion_ratio'] = 0.9

    args.hidden_dim = 2000
    args.emb_dim = 200
    args.p_dim = 200
    config['hidden_dim'] = args.hidden_dim
    config['emb_dim'] = args.emb_dim
    config['p_dim'] = args.p_dim
    config['n_layers_e'] = 4
    config['n_layers_d'] = 2
    config['activation'] = 'leakyrelu'
    # for i in range(1, len(config['AE_arch_e1'])-1):
    #     config['AE_arch_e1'][i] = args.hidden_dim
    #     config['AE_arch_e2'][i] = args.hidden_dim
    # for i in range(1, len(config['AE_arch_d1'])-1):
    #     config['AE_arch_d1'][i] = args.hidden_dim
    #     config['AE_arch_d2'][i] = args.hidden_dim
    # config['AE_arch_e1'][-1] = args.emb_dim
    # config['AE_arch_e2'][-1] = args.latent_dim
    # config['AE_arch_d1'][0] = args.latent_dim
    # config['AE_arch_d2'][0] = args.latent_dim

    config['log_name'] = f"{config['dataset']}_test_20240923_log.txt"
    # config['log_name'] = f"{config['dataset']}_search_20240922_log.txt"
    config['save_path'] = f"figs/"
    # conf = Config(args.dataset)
    # a = conf.dataset

    main(config)
