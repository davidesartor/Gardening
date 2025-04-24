from utils import *
from toy import *
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
from sklearn.model_selection import train_test_split


def plot_growing_trees(sk_IF, val_data, val_labels):

    avg_precision_scores = []
    auc_scores = []

    ordered_indices = sorted_indeces_trees(sk_IF, val_data, val_labels)
    scores = np.cumsum(tree_train, axis=0) / np.arange(1000).reshape(-1, 1)
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
    new_IF = top_k_trees(sk_IF, 100, val_data, val_labels)

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
    plot_growing_trees(sk_IF, val_data, val_labels)


