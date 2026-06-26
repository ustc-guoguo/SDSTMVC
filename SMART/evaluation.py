
import numpy as np
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return acc


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    pur = accuracy_score(y_true, y_voted_labels)

    return pur


def clustering_acc(y_true, y_pred):
    """
    Calculate clustering accuracy and f1-score.

    Parameters
    - y_true: the ground truth.
    - y_pred: the predicted clustering ids.

    Returns
    - acc: clustering accuracy.
    - f1-score: macro f1-score of clustering result.
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print(f"Error! Got num_class1 {num_class1} != numclass2 {numclass2}")
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')

    return acc, f1_macro


def evaluate(y_true, y_pred):
    """
    Evaluate the clustering performance.

    Parameters
    - y_true: The ground truth
    - y_pred: The predicted label

    Return
    - acc: Clustering accuracy.
    - nmi: Normalized mutual information.
    - ari: Adjusted rand index.
    - pur: Purity.
    """
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    pur = purity(y_true, y_pred)

    return acc, nmi, ari, pur


# def clustering(x_list, y):
#     """Get scores of clustering"""
#     n_clusters = np.size(np.unique(y))
#
#     x_final_concat = np.concatenate(x_list[:], axis=1)
#     kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
#                                               init_args={'n_init': 10})
#     y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
#     if np.min(y) == 1:
#         y = y - 1
#     scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)
#
#     ret = {}
#     ret['kmeans'] = scores
#     return ret
#
#
# def calculate_cost_matrix(C, n_clusters):
#     cost_matrix = np.zeros((n_clusters, n_clusters))
#
#     # cost_matrix[i,j] will be the cost of assigning cluster i to label j
#     for j in range(n_clusters):
#         s = np.sum(C[:, j])  # number of examples in cluster i
#         for i in range(n_clusters):
#             t = C[i, j]
#             cost_matrix[j, i] = s - t
#     return cost_matrix
#
#
# def get_cluster_labels_from_indices(indices):
#     n_clusters = len(indices)
#     clusterLabels = np.zeros(n_clusters)
#     for i in range(n_clusters):
#         clusterLabels[i] = indices[i][1]
#     return clusterLabels
#
#
# def get_y_preds(y_true, cluster_assignments, n_clusters):
#     """Computes the predicted labels, where label assignments now
#         correspond to the actual labels in y_true (as estimated by Munkres)
#
#         Args:
#             cluster_assignments: array of labels, outputted by kmeans
#             y_true:              true labels
#             n_clusters:          number of clusters in the dataset
#
#         Returns:
#             a tuple containing the accuracy and confusion matrix,
#                 in that order
#     """
#     confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
#     # compute accuracy based on optimal 1:1 assignment of clusters to labels
#     cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
#     indices = Munkres().compute(cost_matrix)
#     kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
#
#     if np.min(cluster_assignments) != 0:
#         cluster_assignments = cluster_assignments - np.min(cluster_assignments)
#     y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
#     return y_pred.astype(int)
#
#
# def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
#     """Get classification metric"""
#     # confusion matrix
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     # ACC
#     accuracy = metrics.accuracy_score(y_true, y_pred)
#     accuracy = np.round(accuracy, decimals)
#
#     # precision
#     precision = metrics.precision_score(y_true, y_pred, average=average)
#     precision = np.round(precision, decimals)
#
#     # recall
#     recall = metrics.recall_score(y_true, y_pred, average=average)
#     recall = np.round(recall, decimals)
#
#     # F-score
#     f_score = metrics.f1_score(y_true, y_pred, average=average)
#     f_score = np.round(f_score, decimals)
#
#     return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix
#
#
# def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
#     """Get clustering metric"""
#     y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
#
#     classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)
#
#     # AMI
#     ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
#     ami = np.round(ami, decimals)
#     # NMI
#     nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
#     nmi = np.round(nmi, decimals)
#     # ARI
#     ari = metrics.adjusted_rand_score(y_true, y_pred)
#     ari = np.round(ari, decimals)
#
#     return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix
#
#
# def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
#     """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
#         cluster assignments based on input data.
#
#         Args:
#             x: the points with which to perform clustering
#             cluster_obj: a pre-fitted instance of a clustering class
#             ClusterClass: a reference to the sklearn clustering class, necessary
#               if instantiating a new clustering class
#             n_clusters: number of clusters in the dataset, necessary
#                         if instantiating new clustering class
#             init_args: any initialization arguments passed to ClusterClass
#
#         Returns:
#             a tuple containing the label assignments and the clustering object
#     """
#     # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
#     assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
#     cluster_assignments = None
#     if cluster_obj is None:
#         cluster_obj = ClusterClass(n_clusters, **init_args)
#         for _ in range(10):
#             try:
#                 cluster_obj.fit(x)
#                 break
#             except:
#                 print("Unexpected error:", sys.exc_info())
#         else:
#             return np.zeros((len(x),)), cluster_obj
#
#     cluster_assignments = cluster_obj.predict(x)
#     return cluster_assignments, cluster_obj
#
# def acc_rate(y_true, y_pred):
#     """
#     Calculate clustering accuracy.
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     # from sklearn.utils.linear_assignment_ import linear_assignment
#     from scipy.optimize import linear_sum_assignment as linear_assignment
#     ind_row, ind_col = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

# def eval_aligned_detail(P, P_pred, index_mis_aligned, label, device='cuda'):
#     P_gt = np.eye(len(index_mis_aligned)).astype('float32')
#     index = np.argsort(np.argsort(index_mis_aligned))
#     P_gt = P_gt[:, index]
#     P = to_numpy(P)
#     P_pred = to_numpy(P_pred)
#
#     label = label - np.min(label)
#     # 1、看求得的转换矩阵和真实的对齐数量
#     num_acc = P_gt.shape[0] - np.sum(np.abs(P_gt - P_pred)) / 2
#     # 2、看和真实的P为同一个类的数目
#     idx_gt = (P_gt @ index_mis_aligned).astype(int)
#     idx_pred = (P_pred @ index_mis_aligned).astype(int)
#     num_class_acc = (label[idx_gt] == label[idx_pred]).astype(np.int).sum()
#     k = len(np.unique(label))
#     mean, var = eval_acc(P, label[index_mis_aligned], label[idx_gt], k)
#     print('num_acc = %d / %d, num_class_acc = %d / %d, 均值: %.4f, 方差: %.4f' % (num_acc, P_gt.shape[0], num_class_acc, P_gt.shape[0], mean, var))

# def eval_acc(P, shuffle_label, classes, k):
#     n = P.shape[0]
#     total = np.sum(P, axis=1)
#     mask = np.zeros((n, k))
#     for i in range(k):
#         row = shuffle_label == i
#         mask[row, i] = 1
#     one_hot_gt = np.zeros((n, k))
#     one_hot_gt[range(classes.shape[0]), classes] = 1
#     w = np.sum((P @ mask) * one_hot_gt, axis=1)
#     result = w / total
#     mean = np.mean(result)
#     var = np.var(result)
#     return mean, var