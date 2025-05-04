from utils import *
from toy import *

import odds_datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.model_selection import train_test_split

def plot_avg_prc(ap_scores, color = None, label = None, verbose = False):
    mean_ap = np.mean(ap_scores, axis=0)
    std_ap = np.std(ap_scores, axis=0)

    plt.plot(range(1, len(mean_ap) + 1), mean_ap, color=color, label=label)
    plt.fill_between(range(1, len(mean_ap) + 1), mean_ap - std_ap, mean_ap + std_ap,
                        color=color, alpha=0.2)

    plt.xscale('log') # Set x-axis to logarithmic scale
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('Average Precision Score (Avg +/- Std Dev over 10 runs)')
    plt.grid(True, which="both") # Add grid for both major and minor ticks
    plt.legend()

    if verbose:
        print("\n--- Average Precision Scores ---")
        print(f"Max Avg AP: {max(mean_ap):.4f} at {np.argmax(mean_ap) + 1} trees")

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
    n_trees = [100, 300, 1000]

    # Generate data
    for dataset_name in (pbar1:=tqdm(odds_datasets.small_datasets_names)):
        
        all_ap_scores = {n:[] for n in n_trees}
        all_auc_scores = {n:[] for n in n_trees}

        pbar1.set_description(f"{dataset_name} - Loading dataset")
        data, labels = odds_datasets.load(dataset_name)  
        
        for run in (pbar2:=tqdm(range(n_runs), leave=False)):
            current_seed = run
            pbar2.set_description(f"Run {run + 1}/{n_runs}")
            train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(
                data,
                labels,
                val_size=0.2,
                test_size=0.2,
                random_state=current_seed
            )

            for n in n_trees:

                pbar1.set_description(f"{dataset_name} - Training Isolation Forest with {n} trees")
                sk_IF = IsolationForest(n_estimators=n, random_state=current_seed).fit(train_data)

                pbar1.set_description(f"{dataset_name} - Calculating scores with {n} trees")
                ap_scores, auc_scores = score_growing_trees(sk_IF, val_data, val_labels)
    
                all_ap_scores[n].append(ap_scores)
                all_auc_scores[n].append(auc_scores)

        plt.figure(figsize=(10, 7))
        plt.title(f"Average Precision Score vs 100 Trees (Avg/Std Dev) on {dataset_name}")
        for n, c in zip(n_trees, ['red', 'green', 'blue']):
            plot_avg_prc(all_ap_scores[n], color=c, label=f'{n} Trees')

        plt.savefig(f"figures/avg_precision_scores of {dataset_name}.pdf", bbox_inches='tight')

