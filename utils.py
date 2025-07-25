import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
from sklearn.model_selection import train_test_split


def measure(y_true, y_pred):
    return metrics.average_precision_score(y_true, y_pred)

# Function to split data into train, validation, and test sets while keeping the same proportion of labels
def split_data(data, labels, val_size, test_size, random_state=23):
    
    
    for i in range(10):
        
        # First split: separate test set from train+val
        data_train_val, data_test, labels_train_val, labels_test = train_test_split(
            data,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state+i
        )

        # Calculate the proportion of validation set relative to remaining data
        val_proportion = val_size / (1 - test_size)

        # Second split: separate train and val from train+val
        data_train, data_val, labels_train, labels_val = train_test_split(
            data_train_val,
            labels_train_val,
            test_size=val_proportion,
            stratify=labels_train_val,
            random_state=random_state+i
        )
        
        # Check if the split is valid
        not_new_split = (0<labels_test.sum() < len(labels_test)) and (0<labels_train.sum() < len(labels_train)) and (0<labels_val.sum() < len(labels_val))
        if (not_new_split):
            return data_train, labels_train, data_val, labels_val, data_test, labels_test

    raise ValueError("Unable to create a valid split with the given parameters after 10 attempts.")

def plot_prc(y_true, y_pred, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision_score = metrics.average_precision_score(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[5 * 2, 5])

    def ax_plot(ax, x, y, xlabel, ylabel, title=''):
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()

    ax_plot(ax1, fpr, tpr, 'fpr', 'tpr', title=title + " auc: {:.3f}".format(auc))
    ax_plot(ax2, recall, precision, 'recall', 'precision',
            title=title + " average precision: {:.3f}".format(average_precision_score))

    # return auc, average_precision_score

def plot_avg_prc(ap_scores, color=None, label=None, verbose=False, n_runs=50):

    mean_ap = np.mean(ap_scores, axis=0)
    std_ap = np.std(ap_scores, axis=0)

    sem = std_ap / np.sqrt(n_runs)

    x = range(1, len(mean_ap) + 1)

    if color is not None:
        plt.plot(x, mean_ap, color=color, label=label)
        #plt.fill_between(x, mean_ap - std_ap, mean_ap + std_ap, color=color, alpha=0.2)
        plt.fill_between(x, mean_ap - sem, mean_ap + sem, color=color, alpha=0.2)
    else:
        plt.plot(x, mean_ap, label=label)
        #plt.fill_between(x, mean_ap - std_ap, mean_ap + std_ap, alpha=0.2)
        plt.fill_between(x, mean_ap - sem, mean_ap + sem, color=color, alpha=0.2)

    if verbose:
        print("\n--- Average Precision Scores ---")
        print(f"Max Avg AP: {max(mean_ap):.4f} at {np.argmax(mean_ap) + 1} trees")

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

    # average precision for each tree
    ap_tree_train = np.array([measure(val_labels, tree) for tree in tree_train])

    sorted_indices = np.argsort(ap_tree_train)

    return sorted_indices  # from worst to best tree

def compute_tree_scores_rf(forest, data):

    collection_tree_probs = []
    for tree in forest.estimators_:
        probs = tree.predict_proba(data)[:, 1]  # probability of class 1
        collection_tree_probs.append(probs)
    
    return np.array(collection_tree_probs)

def sorted_indices_trees_rf(rf, val_data, val_labels):

    tree_probs = compute_tree_scores_rf(rf, val_data)
    ap_tree = np.array([measure(val_labels, probs) for probs in tree_probs])
    sorted_indices = np.argsort(ap_tree)  
    
    return sorted_indices   # from worst to best tree

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

def score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels):

    avg_precision_scores = []
    auc_scores = []

    n_trees = len(sk_IF.estimators_)

    ordered_indices = sorted_indices_trees(sk_IF, val_data, val_labels)[::-1]  # sort in descending order (best to worst)
    tree_train = compute_tree_anomaly_scores(sk_IF, test_data)  # shape:(n_trees, test_size)
    tree_train_ordered = tree_train[ordered_indices, :]

    scores = np.exp(np.cumsum(np.log(tree_train_ordered), axis=0).T / np.arange(1, n_trees+1))
    scores = scores.T

    # Creating a new IF for every possible number of trees until the maximum has been reached
    for i in range(len(sk_IF.estimators_)):
        y_pred = scores[i]  # Scores using first i+1 trees
        avg_precision_scores.append(measure(test_labels, y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, y_pred))

    return avg_precision_scores, auc_scores

def score_growing_trees_rf(sk_RF, val_data, val_labels, test_data, test_labels):

    avg_precision_scores = []
    auc_scores = []

    n_trees = len(sk_RF.estimators_)


    ordered_indices = sorted_indices_trees_rf(sk_RF, val_data, val_labels)[::-1]
    tree_scores = compute_tree_scores_rf(sk_RF, test_data)
    tree_scores_ordered = tree_scores[ordered_indices, :]

    # Computing cumulative averages
    scores = np.cumsum(tree_scores_ordered, axis=0).T / np.arange(1, n_trees+1)
    scores = scores.T  # Shape: (n_trees, n_samples)

    # Evaluating performance for each number of trees
    for i in range(n_trees):
        y_pred = scores[i]  # averaging probabilities using first i+1 trees
        avg_precision_scores.append(measure(test_labels, y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, y_pred))

    return avg_precision_scores, auc_scores
