# import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def plot_tsne(zs, label, title, save=None):
    tsne = TSNE(n_components=2, init='pca',learning_rate='auto')

    sub_graph_num = len(zs)
    fig, ax = plt.subplots(1,sub_graph_num, figsize=(10, 3.5))

    class_num = max(label)+1
    palette = sns.color_palette("bright", class_num)
    for i in range(sub_graph_num):
        data = tsne.fit_transform(zs[i])
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        sns.scatterplot(data[:,0], data[:,1], ax=ax[i], hue=label, palette=palette)
        ax[i].set_title('V'+str(i))

    plt.suptitle(title, fontsize=14)
    #plt.title('tsne in the {} dataset'.format(name))
    if save:
        plt.savefig(save)
    plt.show()
    return fig


def plot_loss(losses, dim_num=1, save=None, name=None):
    di = {}
    if dim_num == 1:
        se = pd.Series(losses)
        di[0] = se
    else:
        for i in range(len(losses)):
            se = pd.Series(losses[i])
            di[i] = se
    df = pd.DataFrame(di)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df)
    plt.title(name)
    if save:
        plt.savefig(save)
    plt.show()



if __name__ == '__main__':
