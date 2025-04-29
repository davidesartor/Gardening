import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
from sklearn.model_selection import train_test_split


def measure(y_true, y_pred):
    return metrics.average_precision_score(y_true, y_pred)


def plot_prc(y_true, y_pred, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision_score = metrics.average_precision_score(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[5 * 2, 5])

    def ax_plot(ax, x, y, xlabel, ylabel, title=''):
        ax.plot(x, y);
        ax.set_xlabel(xlabel),;
        ax.set_ylabel(ylabel)
        ax.set_title(title);
        ax.grid()

    ax_plot(ax1, fpr, tpr, 'fpr', 'tpr', title=title + " auc: {:.3f}".format(auc))
    ax_plot(ax2, recall, precision, 'recall', 'precision',
            title=title + " average precision: {:.3f}".format(average_precision_score))

    #return auc, average_precision_score


def top_k_trees(sk_IF, k, val_data, val_labels, skip=False, indices=None):
    new_IF = copy.deepcopy(sk_IF)

    if not skip:
        new_IF = copy.deepcopy(sk_IF)
        # compute the anomaly scores for each data sample
        tree_train = compute_tree_anomaly_scores(new_IF, val_data)

        # average precision for each tree
        ap_tree_train = np.array([measure(val_labels, - _tree_train_) for _tree_train_ in tree_train])

        #for i, ap in enumerate(sorted(ap_tree_train), 1):
        #    print(f"Tree {i}: Average Precision = {ap:.3f}")

        top_k_indeces = np.argsort(ap_tree_train)[-k:]

    else:
        new_IF = copy.copy(sk_IF)
        top_k_indeces = indices

    # Fixing the tree features
    new_IF.estimators_ = [new_IF.estimators_[i] for i in top_k_indeces]
    new_IF.estimators_features_ = [new_IF.estimators_features_[i] for i in top_k_indeces]

    # Fixing average path length
    new_IF._average_path_length_per_tree = [
        _average_path_length(tree.tree_.n_node_samples)
        for tree in new_IF.estimators_
    ]

    # Fixing decision path lengths
    new_IF._decision_path_lengths = [
        tree.tree_.compute_node_depths()
        for tree in new_IF.estimators_
    ]

    return new_IF


def compute_tree_anomaly_scores(forest, data):
    collection_tree_anomaly_scores = []

    for tree in forest.estimators_:
        leaves = tree.apply(data)  # index of the leaf in which each sample ends up
        depths = tree.tree_.compute_node_depths()  # returns an array where the index corresponds to the node ID and the value is its depth (root node is depth 0).
        n_samples = tree.tree_.n_node_samples[leaves]  # samples that reached that leaf
        collection_tree_anomaly_scores.append(depths[leaves] + _average_path_length(n_samples))  # computing the score and appending

    norm = _average_path_length(np.array([forest.max_samples_]))  # average path length of max samples

    return 2 ** -(np.array(collection_tree_anomaly_scores) / norm)


def sorted_indices_trees(sk_IF, val_data, val_labels):
    # compute the anomaly scores for each data sample
    tree_train = compute_tree_anomaly_scores(sk_IF, val_data)
    for i in range(len(tree_train)):
        print(f"Tree {i}: {tree_train[i]}")

    # average precision for each tree
    ap_tree_train = np.array([measure(val_labels, tree) for tree in tree_train])

    for i, ap in enumerate(sorted(ap_tree_train), 1):
        print(f"Tree {i}: Average Precision = {ap:.3f}")

    sorted_indices = np.argsort(ap_tree_train)

    return sorted_indices  # from worst to best tree


def pruned_forest(sk_IF, indices):
    new_IF = copy.deepcopy(sk_IF)

    # Fixing the tree features
    new_IF.estimators_ = [new_IF.estimators_[i] for i in indices]
    new_IF.estimators_features_ = [new_IF.estimators_features_[i] for i in indices]

    # Fixing average path length
    new_IF._average_path_length_per_tree = [
        _average_path_length(tree.tree_.n_node_samples)
        for tree in new_IF.estimators_
    ]

    # Fixing decision path lengths
    new_IF._decision_path_lengths = [
        tree.tree_.compute_node_depths()
        for tree in new_IF.estimators_
    ]

    return new_IF
