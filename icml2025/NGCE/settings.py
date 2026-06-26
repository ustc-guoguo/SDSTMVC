import argparse

parser = argparse.ArgumentParser()

# imdb
def imdb():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_imdb', help='model_name')
    parser.add_argument('--train', type=bool, default=False, help='training mode')
    parser.add_argument('--dataset', type=str, default='imdb', help='datasets: acm, dblp, texas, chameleon') # acm_hete_r{:.2f}_{}
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=3, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=500, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--hid_dim', type=int, default=512, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--T0', type=int, default=1400, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.7, help='weight for original structure info')
    args = parser.parse_args()
    return args




# cornell
def cornell():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_cornell', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=3, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=32, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--hid_dim', type=int, default=128, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.2, help='weight for original structure info')
    args = parser.parse_args()
    return args

# wisconsin
def wisconsin():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_wisconsin', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=3, help='aggregation orders')
    parser.add_argument('--T0', type=int, default=1200, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=32, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--hid_dim', type=int, default=128, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.2, help='weight for original structure info')
    args = parser.parse_args()
    return args


def acm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_acm', help='model_name')
    parser.add_argument('--T0', type=int, default=800, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--train', type=bool, default=False, help='training mode')
    parser.add_argument('--dataset', type=str, default='acm', help='datasets: acm, dblp, texas, chameleon') # acm_hete_r{:.2f}_{}
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=1, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--hid_dim', type=int, default=512, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument
    args = parser.parse_args()
    return args



# texas
def texas():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_texas', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=10, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=32, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--T0', type=int, default=1000, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=42, help='')
    parser.add_argument('--hid_dim', type=int, default=128, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=1, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.0, help='weight for original structure info')
    args = parser.parse_args()
    return args


# chameleon
def chameleon():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_chameleon', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=2, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent_dim')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--T0', type=int, default=650, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=42, help='')
    parser.add_argument('--hid_dim', type=int, default=128, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.2, help='weight for original structure info')
    args = parser.parse_args()
    return args

# dblp
def dblp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MVHGC_dblp', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--order', type=int, default=2, help='aggregation orders')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--T0', type=int, default=650, help='first stage epoch')# 580, help='first stage epoch')
    parser.add_argument('--epoch', type=int, default=100000, help='')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=1, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--hid_dim', type=int, default=512, help='hid_dim for recovery')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers for recovery')
    parser.add_argument('--noise_mode', type=int, default=0, help='0:masked, 1:guassian')
    parser.add_argument('--w', type=float, default=0.7, help='weight for original structure info')
    args = parser.parse_args()
    return args

def get_settings(dataset='acm'):
    args_dic = {
        'imdb': imdb(),
        'wisconsin': wisconsin(),
        'acm': acm(),
        'dblp': dblp(),
        'texas': texas(),
        'chameleon': chameleon(),
    }
    return args_dic[dataset]
