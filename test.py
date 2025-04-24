import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
from sklearn.model_selection import train_test_split


# Generating two randomly distributed clusters of n points with n/20 anomalies
def double_cluster_with_generator(seed):
    rng = np.random.default_rng(seed)

    pts = 4000
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

def compute_tree_anomaly_scores(forest, X): #cambiare score
    collection_tree_anomaly_scores = []

    for tree in forest.estimators_:
        leaves = tree.apply(X)  # index of the leaf in which each sample ends up
        depths = tree.tree_.compute_node_depths()   # returns an array where the index corresponds to the node ID and the value is its depth (root node is depth 0).
        n_samples = tree.tree_.n_node_samples[leaves]   # samples that reached that leaf
        collection_tree_anomaly_scores.append(depths[leaves] + _average_path_length(n_samples)) # computing the score and appending

    norm = 1 # average path length di max samples

    return 2**-(np.array(collection_tree_anomaly_scores)/norm)


def measure(y_true, y_pred):

    return metrics.average_precision_score(y_true, y_pred)

def plot_prc(y_true, y_pred, title=''):

    fpr, tpr, thresholds          = metrics.roc_curve(y_true, y_pred)
    auc                           = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision_score       = metrics.average_precision_score(y_true, y_pred)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=[5*2,5])

    def ax_plot(ax,x,y,xlabel,ylabel,title=''):
        ax.plot(x,y);ax.set_xlabel(xlabel),;ax.set_ylabel(ylabel)
        ax.set_title(title);ax.grid()

    ax_plot(ax1,fpr, tpr,'fpr', 'tpr',title=title + " auc: {:.3f}".format(auc))
    ax_plot(ax2,recall,precision, 'recall','precision', title=title + " average precision: {:.3f}".format(average_precision_score))

    #return auc, average_precision_score

def top_k_trees(sk_IF, k, skip=False, indices=None):

    new_IF = copy.deepcopy(sk_IF)

    if not skip:
        new_IF = copy.deepcopy(sk_IF)
        # compute the anomaly scores for each data sample
        tree_train = compute_tree_anomaly_scores(new_IF, val_data)

        # average precision for each tree
        ap_tree_train = np.array([measure(val_labels, - __tree_train__) for __tree_train__ in tree_train])

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

def sorted_indeces_trees(sk_IF):

    # compute the anomaly scores for each data sample
    tree_train = compute_tree_anomaly_scores(sk_IF, val_data)

    # average precision for each tree
    ap_tree_train = np.array([measure(val_labels, - __tree_train__) for __tree_train__ in tree_train])

    #for i, ap in enumerate(sorted(ap_tree_train), 1):
    #    print(f"Tree {i}: Average Precision = {ap:.3f}")

    top_k_indeces = np.argsort(ap_tree_train)

    return top_k_indeces

def pruned_forest(sk_IF, indices):

    new_IF = copy.deepcopy(sk_IF)

    # Fixing the tree features
    new_IF.estimators_ = [new_IF.estimators_[i] for i in indeces]
    new_IF.estimators_features_ = [new_IF.estimators_features_[i] for i in indeces]

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


# Function to split data into train, validation, and test sets while keeping the same proportion of labels
def split_data(data, labels, val_size, test_size, random_state=23):

    # First split: separate test set from train+val
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Calculate the proportion of validation set relative to remaining data
    val_proportion = val_size / (1 - test_size)

    # Second split: separate train and val from train+val
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_train_val,
        labels_train_val,
        test_size=val_proportion,
        stratify=labels_train_val,
        random_state=random_state
    )

    return data_train, labels_train, data_val, labels_val, data_test, labels_test

def plot_growing_trees(sk_IF):

    avg_precision_scores = []
    auc_scores = []

    ordered_indices = sorted_indeces_trees(sk_IF)

    # Creating a new IF for every possible number of trees until the maximum has been reached
    for i in range(1, len(sk_IF.estimators_)+1):
        print("Iteration: ", i)
        indices = ordered_indices[-i:]
        i_th_forest = pruned_forest(sk_IF, indices)
        y_pred = i_th_forest.score_samples(test_data)
        avg_precision_scores.append(measure(test_labels, -y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, -y_pred))

    # Plotting the average precision scores
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, len(sk_IF.estimators_)+1), avg_precision_scores)
    plt.xlabel('Number of Trees')
    plt.ylabel('Average Precision Score')
    plt.title('Average Precision Score vs Number of Trees')
    plt.grid(True)
    plt.show()
    print("Maximum average precision score: ", max(avg_precision_scores), " with: ", avg_precision_scores.index(max(avg_precision_scores))+1, " trees")

    # Plotting the AUC scores
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, len(sk_IF.estimators_)+1), auc_scores)
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC Score')
    plt.title('AUC Score vs Number of Trees')
    plt.grid(True)
    plt.show()
    print("Maximum AUC score: ", max(auc_scores), " with: ", auc_scores.index(max(auc_scores))+1, " trees")


if __name__ == '__main__':
    # Generate data
    data, labels = double_cluster_with_generator(1)

    # Split into 60% train, 20% val, 20% test
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(
        data,
        labels,
        val_size=0.2,
        test_size=0.2,
        random_state=1
    )

    # Unsupervised training
    sk_IF = IsolationForest(n_estimators=1000, random_state=0, ).fit(train_data)

    # Scores using the training set
    y_pred_train = sk_IF.score_samples(train_data)
    plot_prc(train_labels, -y_pred_train, "Training set")

    # Scores using the test set
    y_pred = sk_IF.score_samples(test_data)
    plot_prc(test_labels, -y_pred, "Test set")


    # Plot the anomaly score
    plt.figure(figsize=(5, 5))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=y_pred)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    #plt.show()

    #################################################
    # Creating a new IF with the top k trees
    new_IF = top_k_trees(sk_IF, 100)

    # Scores using the training set
    y_pred_train_new = new_IF.score_samples(train_data)
    plot_prc(train_labels, -y_pred_train_new, "Training set")

    # Scores using the test set
    y_pred_new = new_IF.score_samples(test_data)
    plot_prc(test_labels, -y_pred_new, "Test set")

    # Plot the new anomaly score
    plt.figure(figsize=(5, 5))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=y_pred_new)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.show()

    # Plotting the average precision and AUC scores for all possible numbers of best trees
    plot_growing_trees(sk_IF)



 # print(tree_train.shape)
 # ################################################
 # # ordinare scores
 # scores = np.cumsum(tree_train, axis=0) / np.arange(1000).reshape(-1, 1)
 # 2 ** -scores
 # # trick: esponenziale media = media gemotrica degli esponenziali
 # # media geometrica --> exp (media log)

