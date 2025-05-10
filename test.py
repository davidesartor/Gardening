from utils import sorted_indices_trees, compute_tree_anomaly_scores, measure, split_data
from toy import double_cluster_with_generator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics

def score_growing_trees(sk_IF, val_data, val_labels):

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

if __name__ == '__main__':

    n_runs = 10
    all_ap_scores_100 = []
    all_auc_scores_100 = []
    all_ap_scores_300 = []
    all_auc_scores_300 = []
    all_ap_scores_1000 = []
    all_auc_scores_1000 = []

    # Generate data
    print(f"Performing {n_runs} runs...")
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        current_seed = run

        data, labels = double_cluster_with_generator(current_seed)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(
            data,
            labels,
            val_size=0.2,
            test_size=0.2,
            random_state=current_seed
        )

        print("Training Isolation Forest models...")
        sk_IF_100 = IsolationForest(n_estimators=100, random_state=current_seed).fit(train_data)
        sk_IF_300 = IsolationForest(n_estimators=300, random_state=current_seed).fit(train_data)
        sk_IF_1000 = IsolationForest(n_estimators=1000, random_state=current_seed).fit(train_data)
        print("Training complete.")

        print("Calculating scores...")
        ap_scores_100, auc_scores_100 = score_growing_trees(sk_IF_100, val_data, val_labels)
        ap_scores_300, auc_scores_300 = score_growing_trees(sk_IF_300, val_data, val_labels)
        ap_scores_1000, auc_scores_1000 = score_growing_trees(sk_IF_1000, val_data, val_labels)
        print("Score calculation complete.")

        all_ap_scores_100.append(ap_scores_100)
        all_auc_scores_100.append(auc_scores_100)
        all_ap_scores_300.append(ap_scores_300)
        all_auc_scores_300.append(auc_scores_300)
        all_ap_scores_1000.append(ap_scores_1000)
        all_auc_scores_1000.append(auc_scores_1000)

    print("\nAll runs complete. Calculating averages and standard deviations...")


    # Calculate mean and standard deviation for each run
    mean_ap_100 = np.mean(all_ap_scores_100, axis=0)
    std_ap_100 = np.std(all_ap_scores_100, axis=0)
    mean_auc_100 = np.mean(all_auc_scores_100, axis=0)
    std_auc_100 = np.std(all_auc_scores_100, axis=0)

    mean_ap_300 = np.mean(all_ap_scores_300, axis=0)
    std_ap_300 = np.std(all_ap_scores_300, axis=0)
    mean_auc_300 = np.mean(all_auc_scores_300, axis=0)
    std_auc_300 = np.std(all_auc_scores_300, axis=0)

    mean_ap_1000 = np.mean(all_ap_scores_1000, axis=0)
    std_ap_1000 = np.std(all_ap_scores_1000, axis=0)
    mean_auc_1000 = np.mean(all_auc_scores_1000, axis=0)
    std_auc_1000 = np.std(all_auc_scores_1000, axis=0)

    plt.figure(figsize=(10, 7))

    plt.plot(range(1, len(sk_IF_100.estimators_) + 1), mean_ap_100, color='blue', label='100 Trees (Avg AP)')
    plt.fill_between(range(1, len(sk_IF_100.estimators_) + 1), mean_ap_100 - std_ap_100, mean_ap_100 + std_ap_100,
                     color='blue', alpha=0.2)

    plt.plot(range(1, len(sk_IF_300.estimators_) + 1), mean_ap_300, color='red', label='300 Trees (Avg AP)')
    plt.fill_between(range(1, len(sk_IF_300.estimators_) + 1), mean_ap_300 - std_ap_300, mean_ap_300 + std_ap_300,
                     color='red', alpha=0.2)

    plt.plot(range(1, len(sk_IF_1000.estimators_) + 1), mean_ap_1000, color='green', label='1000 Trees (Avg AP)')
    plt.fill_between(range(1, len(sk_IF_1000.estimators_) + 1), mean_ap_1000 - std_ap_1000, mean_ap_1000 + std_ap_1000,
                     color='green', alpha=0.2)

    plt.xscale('log') # Set x-axis to logarithmic scale
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('Average Precision Score (Avg +/- Std Dev over 10 runs)')
    plt.title('Average Precision Score vs Number of Trees (Avg/Std Dev)')
    plt.grid(True, which="both") # Add grid for both major and minor ticks
    plt.legend()

    print("\n--- Average Precision Scores (Avg over 10 runs) ---")
    print(f"Max Avg AP (100 Trees): {max(mean_ap_100):.4f} at {np.argmax(mean_ap_100) + 1} trees")
    print(f"Max Avg AP (300 Trees): {max(mean_ap_300):.4f} at {np.argmax(mean_ap_300) + 1} trees")
    print(f"Max Avg AP (1000 Trees): {max(mean_ap_1000):.4f} at {np.argmax(mean_ap_1000) + 1} trees")

    plt.figure(figsize=(10, 7))

    plt.plot(range(1, len(sk_IF_100.estimators_) + 1), mean_auc_100, color='blue', label='100 Trees (Avg AUC)')
    plt.fill_between(range(1, len(sk_IF_100.estimators_) + 1), mean_auc_100 - std_auc_100, mean_auc_100 + std_auc_100,
                     color='blue', alpha=0.2)

    plt.plot(range(1, len(sk_IF_300.estimators_) + 1), mean_auc_300, color='red', label='300 Trees (Avg AUC)')
    plt.fill_between(range(1, len(sk_IF_300.estimators_) + 1), mean_auc_300 - std_auc_300, mean_auc_300 + std_auc_300,
                     color='red', alpha=0.2)

    plt.plot(range(1, len(sk_IF_1000.estimators_) + 1), mean_auc_1000, color='green', label='1000 Trees (Avg AUC)')
    plt.fill_between(range(1, len(sk_IF_1000.estimators_) + 1), mean_auc_1000 + std_auc_1000,
                     mean_auc_1000 - std_auc_1000, color='green', alpha=0.2)

    plt.xscale('log') # Set x-axis to logarithmic scale
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('AUC Score (Avg +/- Std Dev over 10 runs)')
    plt.title('AUC Score vs Number of Trees (Avg/Std Dev)')
    plt.grid(True, which="both") # Add grid for both major and minor ticks
    plt.legend()

    print("\n--- AUC Scores (Avg over 10 runs) ---")
    print(f"Max Avg AUC (100 Trees): {max(mean_auc_100):.4f} at {np.argmax(mean_auc_100) + 1} trees")
    print(f"Max Avg AUC (300 Trees): {max(mean_auc_300):.4f} at {np.argmax(mean_auc_300) + 1} trees")
    print(f"Max Avg AUC (1000 Trees): {max(mean_auc_1000):.4f} at {np.argmax(mean_auc_1000) + 1} trees")

    plt.show()






