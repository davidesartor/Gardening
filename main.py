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


def measure(y_true, y_pred):

    return metrics.average_precision_score(y_true, y_pred)


def compute_tree_anomaly_scores(forest, X):
    collection_tree_anomaly_scores = []

    for tree in forest.estimators_:

        leaves_index = tree.apply(X) # index of the leaf reached by each sample
        node_indicator = tree.decision_path(X) # sparse matrix where each row corresponds to a data point and each column corresponds to a node in the tree
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index] # number of samples that reached that node

        # summing path legth to the average path length and subtracting one (remove root node)
        tree_anomaly_scores = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        collection_tree_anomaly_scores.append(tree_anomaly_scores)

    return np.array(collection_tree_anomaly_scores)

if __name__ == '__main__':
    data_train, labels_train = double_cluster_with_generator(1)
    plt.show()

    # Unsupervised training

    sk_IF = IsolationForest(random_state=0).fit(data_train)
    y_pred = sk_IF.score_samples(data_train)

    # Plot the anomaly score
    plt.figure(figsize=(5, 5))
    plt.scatter(data_train[:, 0], data_train[:, 1], c=y_pred)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.show()

    # compute the anomaly scores for each data
    tree_train = compute_tree_anomaly_scores(sk_IF, data_train)

    # average precision for each tree
    ap_tree_train = np.array([measure(labels_train, - __tree_train__) for __tree_train__ in tree_train])

    for i, ap in enumerate(sorted(ap_tree_train), 1):
        print(f"Tree {i}: Average Precision = {ap:.3f}")

    # histogram of average precisions of the trees
    plt.hist(ap_tree_train, bins=20, edgecolor='black')
    plt.title('Histogram of Tree Average Precision Scores')
    plt.xlabel('Average Precision')
    plt.ylabel('Number of Trees')
    plt.axvline(np.mean(ap_tree_train), color='r', linestyle='--', label=f'Mean AP = {np.mean(ap_tree_train):.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()
