from utils import *
from toy import *
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
from sklearn.model_selection import train_test_split


def plot_growing_trees_slow(sk_IF, val_data, val_labels):

    avg_precision_scores = []
    auc_scores = []

    ordered_indices = sorted_indices_trees(sk_IF, val_data, val_labels)

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
    #plt.show()
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

    # Plotting the average precision and AUC scores for all possible numbers of best trees
    plot_growing_trees_slow(sk_IF, val_data, val_labels)

