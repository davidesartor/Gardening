from utils import *
from toy import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.model_selection import train_test_split


def plot_growing_trees(sk_IF, val_data, val_labels):

    avg_precision_scores = []
    auc_scores = []

    # a(x_j) = 2 ** -E_i[h_i(x_j)]/c
    # tt[i,j] = 2 ** -h_i(x_j)/c
    #  exp(somma(log(tt))/n_trees)
    n_trees = len(sk_IF.estimators_)

    ordered_indices = sorted_indices_trees(sk_IF, val_data, val_labels)[::-1]  # sort in descending order (best to worst)
    tree_train = compute_tree_anomaly_scores(sk_IF, test_data)  # shape:(n_trees, test_size)
    tree_train_ordered = tree_train[ordered_indices, :]

    scores = np.exp(np.cumsum(np.log(tree_train_ordered), axis=0).T / np.arange(1, n_trees+1))
    scores = scores.T

    # Creating a new IF for every possible number of trees until the maximum has been reached
    for i in range(len(sk_IF.estimators_)):
        y_pred = scores[i]  # Scores using first i+1 trees
        if i == sk_IF.n_estimators - 1:
            print("y_predfor: ", y_pred)
        avg_precision_scores.append(measure(test_labels, y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, y_pred))


    # Plotting the average precision scores
    plt.figure(figsize=(5, 5))
    plt.scatter(range(1, len(sk_IF.estimators_)+1), avg_precision_scores, linewidths=0.2)
    plt.xlabel('Number of Trees')
    plt.ylabel('Average Precision Score')
    plt.title('Average Precision Score vs Number of Trees')
    plt.grid(True)
    print("Maximum average precision score: ", max(avg_precision_scores), " with: ", avg_precision_scores.index(max(avg_precision_scores))+1, " trees")

    # Plotting the AUC scores
    plt.figure(figsize=(5, 5))
    plt.scatter(range(1, len(sk_IF.estimators_)+1), auc_scores, linewidths=0.2)
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
    sk_IF = IsolationForest(n_estimators=100, random_state=0, ).fit(train_data)

    # Plotting the average precision and AUC scores for all possible numbers of best trees
    #sorted_indices_trees(sk_IF, val_data, val_labels)
    plot_growing_trees(sk_IF, val_data, val_labels)

    y_pred = sk_IF.score_samples(test_data)
    print("y_pred: ", -y_pred)
    print(measure(test_labels, -y_pred))


