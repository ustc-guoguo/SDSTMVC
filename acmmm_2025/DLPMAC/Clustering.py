import torch
import numpy as np
import torch.nn
from utils import cosineSimilartydis
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from munkres import Munkres
import sys
import logging

def tiny_infer(model, device, all_data, all_label_X, all_label_Y):
    model.eval()
    align_out0 = []
    align_out1 = []
    sort_value=[]
    class_labels_cluster = []
    len_alldata0 = all_data[0].shape[1]
    len_alldata1 = all_data[1].shape[1]

    len_map=max(len_alldata0, len_alldata1)
    align_labels = torch.zeros(len_map)
    if len_alldata0 > len_alldata1:
        labels = all_label_Y
        long_labels=all_label_X
        test_num = len_alldata1
        long_num= len_alldata0
    else:
        labels = all_label_X
        long_labels = all_label_Y
        test_num = len_alldata0
        long_num = len_alldata1
    labels = torch.from_numpy(labels)
    with torch.no_grad():
        x0, x1, labels = all_data[0].to(device), all_data[1].to(device), labels.to(device)
        x0 = x0.view(x0.size()[0], -1).T
        x1 = x1.view(x1.size()[0], -1).T
        h0, h1,_,_ = model(x0, x1)
        if len_alldata0 > len_alldata1:
            C = cosineSimilartydis(h0, h1).T
            C_temp=C.clone()
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                sort_value.append(C_temp[i, idx[0]])
                C[:, idx[0]] = C[:, idx[0]]+0.1
                # C[:, idx[0]] = float("inf")
                align_out0.append((h1[i, :].cpu()).numpy())
                align_out1.append((h0[idx[0], :].cpu()).numpy())#它和align0维度一样变小了
                if all_label_Y[i] == all_label_X[idx[0]]:
                    align_labels[i] = 1
        else:
            C = cosineSimilartydis(h0, h1)
            C_temp = C.clone()
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                sort_value.append(C_temp[i, idx[0]])
                C[:, idx[0]] = C[:, idx[0]] + 0.1
                # C[:, idx[0]] = float("inf")
                align_out0.append((h0[i, :].cpu()).numpy())
                align_out1.append((h1[idx[0], :].cpu()).numpy())
                if all_label_X[i] == all_label_Y[idx[0]]:
                    align_labels[i] = 1

        class_labels_cluster.extend(labels.cpu().numpy())
#
    count = torch.sum(align_labels)
    # print(test_num,'testnum')
    inference_acc = count.item() / test_num
    # print(inference_acc)
    # print(np.shape(align_out1))
    # return np.array(alignre0), np.array(alignre1), np.array(class_labels_cluster), inference_acc
    return np.array(align_out0), np.array(align_out1), np.array(class_labels_cluster), inference_acc
def Clustering(x_list, y):
    # logging.info('******** Clustering ********')
    n_clusters = np.size(np.unique(y))

    # np.random.seed(1)

    x_final_concat = np.concatenate(x_list[:], axis=1)
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})
    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
    # print(y_preds)
    # print(y)
    if np.min(y) == 1:
        y = y - 1
    scores, _ ,accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity= clustering_metric(y, kmeans_assignments, n_clusters)

    ret = {}
    ret['kmeans'] = scores
    return y_preds, ret,accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity

def get_y_preds(y_true, cluster_assignments, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    '''
    Using either a newly instantiated ClusterClass or a provided
    cluster_obj, generates cluster assignments based on input data

    x:              the points with which to perform clustering
    cluster_obj:    a pre-fitted instance of a clustering class
    ClusterClass:   a reference to the sklearn clustering class, necessary
                    if instantiating a new clustering class
    n_clusters:     number of clusters in the dataset, necessary
                    if instantiating new clustering class
    init_args:      any initialization arguments passed to ClusterClass

    returns:    a tuple containing the label assignments and the clustering object
    '''
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def clustering_metric(y_true, y_pred, n_clusters, verbose=False, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)
    accuracy = metrics.accuracy_score(y_true, y_pred_ajusted)
    accuracy = np.round(accuracy, decimals)
    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred_ajusted)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred_ajusted)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred_ajusted)
    ari = np.round(ari, decimals)
    #fscore
    f_score = metrics.f1_score(y_true, y_pred_ajusted, average='macro')
    f_score = np.round(f_score, decimals)
    f_score2 = metrics.f1_score(y_true, y_pred_ajusted, average='weighted')
    f_score2 = np.round(f_score2, decimals)
    # precision
    precision = metrics.precision_score(y_true, y_pred_ajusted, average='macro')
    precision = np.round(precision, decimals)
    precision2 = metrics.precision_score(y_true, y_pred_ajusted, average='weighted')
    precision2 = np.round(precision2, decimals)
    # recall
    recall = metrics.recall_score(y_true, y_pred_ajusted, average='macro')
    recall = np.round(recall, decimals)
    # Purity
    purity = Purity(y_true, y_pred_ajusted)
    purity = np.round(purity, decimals)
    # print(accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity,"zb")
    # if verbose:
    #     logging.info('AMI: {}, NMI: {}, ARI: {}'.format(ami, nmi, ari))
    # return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix,accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity
    return dict({'ACC': accuracy,'AMI': ami, 'NMI': nmi, 'ARI': ari, 'F1': f_score, 'F2': f_score2, 'PRE': precision, 'PRE2': precision2, 'REC': recall, 'PUR': purity}), confusion_matrix, accuracy, nmi, ari, f_score, f_score2, precision, precision2, recall, purity
def Purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)

def classification_metric(y_true, y_pred, average='macro', verbose=False, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    if verbose:
        # print('Confusion Matrix')
        # print(confusion_matrix)
        logging.info('accuracy: {}, precision: {}, recall: {}, f_measure: {}'.format(accuracy, precision, recall, f_score))
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix