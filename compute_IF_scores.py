import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed

# Assumes these functions are in your utils.py file
from utils import split_data, score_growing_trees

def compute_and_save_scores(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):
    
    # Initialize dictionaries to store scores for each number of trees
    all_ap_scores = {n: [] for n in n_trees}
    all_auc_scores = {n: [] for n in n_trees}

    # Define processing function for single run
    def process_single_run(run):
        current_seed = run
        train_data, _, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )
        
        run_ap_scores = {n: [] for n in n_trees}
        run_auc_scores = {n: [] for n in n_trees}
        
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

    # Convert lists of lists to 2D NumPy arrays: shape (n_runs, n) for each n
    ap_scores_dict = {n: np.array(all_ap_scores[n]) for n in n_trees}
    auc_scores_dict = {n: np.array(all_auc_scores[n]) for n in n_trees}

    os.makedirs(save_dir, exist_ok=True)

    # Save both dictionaries in a single .npz file
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
    main_save_dir = "results_if"

    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    for dataset_name in tqdm(odds_datasets.datasets_names, desc="Processing datasets"):
        data, labels = odds_datasets.load(dataset_name)

        for val_size in val_sizes:
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            plots_dir = os.path.join(val_size_dir, "plots") # This plots_dir is for the 'by_val_size' plots
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            try:
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