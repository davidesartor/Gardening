import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed

# Assumes these functions are in your utils.py file
from utils import compute_tree_anomaly_scores, measure, split_data

def compute_bruteforce_points(dataset_name, data, labels, n_trees, n_runs, bf_values, max_iter, val_size, test_size,
                              save_dir):
    all_ap_bf_scores = {n: [] for n in n_trees}
    all_max_ap_bf_scores = {n: [] for n in n_trees}

    def process_single_bf_run(run_seed, n_estimators):

        np.random.seed(run_seed)

        train_data, _, val_data, val_label, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=run_seed
        )

        sk_IF = IsolationForest(n_estimators=n_estimators, random_state=run_seed, n_jobs=-1).fit(train_data)
        tree_train = compute_tree_anomaly_scores(sk_IF, val_data).astype(np.float32)  # (n_estimators, n_test)
        tree_test = compute_tree_anomaly_scores(sk_IF, test_data).astype(np.float32)

        ap_bf_scores_for_run = []
        max_ap_bf_scores_for_run = []

        # Precompute random indices
        random_indices = {
            x: [np.random.choice(n_estimators, x, replace=False) for _ in range(max_iter)]
            for x in bf_values if x <= n_estimators
        }

        for x in bf_values:
            if x <= n_estimators:
                # Use list comprehension
                scores_val = [np.exp(np.mean(np.log(tree_train[idx, :]), axis=0)) for idx in random_indices[x]]
                scores_test = [np.exp(np.mean(np.log(tree_test[idx, :]), axis=0)) for idx in random_indices[x]]
                ap_val_list_for_x = [measure(val_label, s) for s in scores_val]
                ap_test_list_for_x = [measure(test_labels, s) for s in scores_test]
                
                max_index = np.argmax(np.array(ap_val_list_for_x))

                ap_bf_scores_for_run.append(np.mean(ap_test_list_for_x))
                max_ap_bf_scores_for_run.append(ap_test_list_for_x[max_index])
            else:
                ap_bf_scores_for_run.append(np.nan)
                max_ap_bf_scores_for_run.append(np.nan)

        return ap_bf_scores_for_run, max_ap_bf_scores_for_run

    # Top-level parallelism over (run, n)
    results = Parallel(n_jobs=-1)(
        delayed(process_single_bf_run)(run, n)
        for run in tqdm(range(n_runs), desc=f"{dataset_name} - Brute-Force Runs", leave=False)
        for n in n_trees
    )

    # Reshape results into dictionary format
    result_idx = 0
    for _ in range(n_runs):
        for n in n_trees:
            ap_scores, max_ap_scores = results[result_idx]
            all_ap_bf_scores[n].append(ap_scores)
            all_max_ap_bf_scores[n].append(max_ap_scores)
            result_idx += 1

    # Convert to numpy arrays
    ap_bf_scores_dict = {n: np.array(all_ap_bf_scores[n]) for n in n_trees}
    max_ap_bf_scores_dict = {n: np.array(all_max_ap_bf_scores[n]) for n in n_trees}

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"{dataset_name}_if_bf_scores.npz"),
        ap_scores=ap_bf_scores_dict,
        max_ap_scores=max_ap_bf_scores_dict,
        allow_pickle=True
    )

if __name__ == "__main__":
    # Parameters
    n_runs = 10
    n_trees = [100, 300, 1000]
    val_sizes = [0.01, 0.05, 0.1, 0.2]
    test_size = 0.2
    max_iter = 5000
    bf_values = [10, 50, 100, 300, 800]
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
                print(f"Computing brute-force scores for {dataset_name}, val_size={val_size}...")
                compute_bruteforce_points(
                    dataset_name=dataset_name,
                    data=data,
                    labels=labels,
                    n_trees=n_trees,
                    n_runs=n_runs,
                    bf_values=bf_values,
                    max_iter=max_iter,
                    val_size=val_size,
                    test_size=test_size,
                    save_dir=data_dir
                )
            except ValueError as e:
                print(f"Error processing {dataset_name} with val_size {val_size}: {e}")
                continue