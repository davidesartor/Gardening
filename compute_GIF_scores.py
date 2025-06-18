import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed

# Assumes these functions are in your utils.py file
from utils import compute_tree_anomaly_scores, measure, split_data

def score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels):
    avg_precision_scores = []
    auc_scores = []

    n_trees = len(sk_IF.estimators_)

    # Get tree indices using greedy approach
    ordered_indices = sorted_indices_trees_greedy(sk_IF, val_data, val_labels)
    tree_train = compute_tree_anomaly_scores(sk_IF, test_data)  # Shape: (n_trees, n_test_samples)
    tree_train_ordered = tree_train[ordered_indices, :]  # Reorder by greedy selection

    # Compute cumulative geometric mean of anomaly scores
    scores = np.exp(np.cumsum(np.log(tree_train_ordered), axis=0).T / np.arange(1, n_trees+1))
    scores = scores.T  # Shape: (n_trees, n_test_samples)

    # Evaluate performance for each number of trees
    for i in range(n_trees):
        y_pred = scores[i]  # Scores using first i+1 trees
        avg_precision_scores.append(measure(test_labels, y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, y_pred))

    return avg_precision_scores, auc_scores

def sorted_indices_trees_greedy(sk_IF, val_data, val_labels):
    n_trees = len(sk_IF.estimators_)
    tree_scores = compute_tree_anomaly_scores(sk_IF, val_data)
    
    # Find best initial tree
    ap_scores = Parallel(n_jobs=2)(delayed(measure)(val_labels, tree) for tree in tree_scores)
    best_tree_idx = np.argmax(ap_scores)
    selected_indices = [best_tree_idx]
    available_indices = np.ones(n_trees, dtype=bool)
    available_indices[best_tree_idx] = False
    cumulative_log_scores = np.log(tree_scores[best_tree_idx])

    # Greedy selection over available candidates
    for _ in tqdm(range(1, n_trees), leave=False):
        candidate_indices = np.where(available_indices)[0]
        if len(candidate_indices) == 0:
            break
        
        # Compute combined scores only for candidates
        temp_combined = np.exp(
            (cumulative_log_scores + np.log(tree_scores[candidate_indices])) / 
            (len(selected_indices) + 1)
        )
        
        ap_scores = Parallel(n_jobs=-1)(delayed(measure)(val_labels, score) for score in temp_combined)
        best_candidate_idx = np.argmax(ap_scores)
        best_idx = candidate_indices[best_candidate_idx]
        
        selected_indices.append(best_idx)
        available_indices[best_idx] = False
        cumulative_log_scores += np.log(tree_scores[best_idx])
        
    return np.array(selected_indices)

def compute_and_save_scores(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):

    # dictionaries to store scores
    all_ap_scores = {n: [] for n in n_trees}
    all_auc_scores = {n: [] for n in n_trees}

    def process_single_run(run):
        current_seed = run

        train_data, _, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )

        run_ap_scores = {n: [] for n in n_trees}
        run_auc_scores = {n: [] for n in n_trees}

        # computing scores for each number of trees
        for n in n_trees:

            sk_IF = IsolationForest(n_estimators=n, random_state=current_seed).fit(train_data)
            ap_scores, auc_scores = score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels)

            run_ap_scores[n] = ap_scores
            run_auc_scores[n] = auc_scores

        return run_ap_scores, run_auc_scores

    # Parallelize runs using joblib
    results = Parallel(n_jobs=10)(
        delayed(process_single_run)(run)
        for run in tqdm(range(n_runs), desc=f"{dataset_name} - Runs", leave=False)
    )

    # Aggregate results from all runs
    for run_ap, run_auc in results:
        for n in n_trees:
            all_ap_scores[n].append(run_ap[n])
            all_auc_scores[n].append(run_auc[n])

    # converting lists of lists to 2D NumPy arrays (n_runs, n) 
    ap_scores_dict = {n: np.array(all_ap_scores[n]) for n in n_trees}
    auc_scores_dict = {n: np.array(all_auc_scores[n]) for n in n_trees}

    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, f"{dataset_name}_if_scores.npz"),
        ap_scores=ap_scores_dict,
        auc_scores=auc_scores_dict,
        allow_pickle=True
    )

if __name__ == "__main__":
    # Parameters
    n_runs = 10
    n_trees = [100, 300, 1000]
    val_sizes = [0.01, 0.05, 0.1, 0.2]
    test_size = 0.2
    main_save_dir = "results_greedy_if"

    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    for dataset_name in tqdm(odds_datasets.small_datasets_names, desc="Processing datasets"):
        data, labels = odds_datasets.load(dataset_name)
        for val_size in val_sizes:
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            try:
                print(f"Computing greedy scores for {dataset_name}, val_size={val_size}...")
                compute_and_save_scores(
                    dataset_name=dataset_name,
                    data=data,
                    labels=labels,
                    n_trees=n_trees,
                    n_runs=n_runs,
                    val_size=val_size,
                    test_size=test_size,
                    save_dir=data_dir
                )
            except ValueError as e:
                print(f"Error processing {dataset_name} with val_size {val_size}: {e}")
                continue