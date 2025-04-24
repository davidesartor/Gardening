import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length


# Generating two randomly distributed clusters of n points with n/20 anomalies
def double_cluster_with_generator(seed):
    rng = np.random.default_rng(seed)

    pts = 500
    std = 0.1
    step = 0.4
    sx_cluster = rng.standard_normal((pts, 2)) * std + step
    dx_cluster = rng.standard_normal((pts, 2)) * std - step

    anomaly = (rng.random((round(pts / 20), 2)) - 0.5) * 2

    data = np.vstack([sx_cluster, dx_cluster, anomaly])
    labels = (np.linalg.norm(data + step, axis=1) > std * 3) & (np.linalg.norm(data - step, axis=1) > std * 3)

    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], c=1 - labels, cmap='Set1')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xticks([])
    plt.yticks([])

    return data, labels

def plot_prc(fpr, tpr,auc,recall,precision,average_precision_score):

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=[5*2,5])

    def ax_plot(ax,x,y,xlabel,ylabel,title=''):
        ax.plot(x,y); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title);ax.grid()

    ax_plot(ax1,fpr, tpr,'fpr', 'tpr',title="auc: {:.3f}".format(auc))
    ax_plot(ax2,recall,precision, 'recall','precision', title="average precision: {:.3f}".format(average_precision_score))

def measure(y_true, y_pred, plot = False):

    # apply metrics
    fpr, tpr, thresholds          = metrics.roc_curve(y_true, y_pred)
    auc                           = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision_score       = metrics.average_precision_score(y_true, y_pred)

    if plot == True:
        plot_prc(fpr, tpr,auc,recall,precision,average_precision_score)
    else:
        return average_precision_score


def compute_tree_anomaly_scores(forest, X):
    """
    Compute the score of each samples in X going through the extra trees.
    Parameters
    ----------
    X : array-like or sparse matrix
        Data matrix.
    subsample_features : bool
        Whether features should be subsampled.
    """
    n_samples = X.shape[0]

    depths = np.zeros(n_samples, order="f")

    collection_tree_anomaly_scores = []

    for tree in forest.estimators_:
        leaves_index = tree.apply(X)
        node_indicator = tree.decision_path(X)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        tree_anomaly_scores = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf)- 1.0)

        depths += tree_anomaly_scores

        collection_tree_anomaly_scores.append(tree_anomaly_scores)

    denominator = len(forest.estimators_) * _average_path_length([forest.max_samples_])
    scores = 2 ** (
        # For a single training sample, denominator and depth are 0.
        # Therefore, we set the score manually to 1.
        -np.divide(depths, denominator, out=np.ones_like(depths), where=denominator != 0)
    )
    return scores, np.array(collection_tree_anomaly_scores)

if __name__ == '__main__':
    data_train, labels_train = double_cluster_with_generator(1)
    plt.show()

    # Unsupervised training

    sk_IF = IsolationForest(random_state=0).fit(data_train)
    y_pred = sk_IF.score_samples(data_train)
    # y_grid_pred = sk_IF.score_samples(data_grid)

    # Plot the anomaly score
    plt.figure(figsize=(5, 5))
    plt.scatter(data_train[:, 0], data_train[:, 1], c=y_pred)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.show()
    # plt.contour(x_grid, y_grid, y_grid_pred.reshape(n, n), levels=25)
    # measure(labels_train, -y_pred, plot=True)


    # compute the anomaly scores for each data
    sklean_scores, tree_train = compute_tree_anomaly_scores(sk_IF, data_train)

    for i,s in enumerate(tree_train, 1):
        print(i,s)

    # average precision for each tree
    ap_tree_train = np.array([measure(labels_train, - __tree_train__) for __tree_train__ in tree_train])

    # histogram of average precisions of the trees
    _ = plt.hist(ap_tree_train)
    plt.title('Histogram of the tree average precision')
    plt.grid(True)
    plt.show()
