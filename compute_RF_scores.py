from utils import split_data, measure

import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

def compute_and_save_rf_scores(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):
    all_ap_scores = {n: [] for n in n_trees}
    all_auc_scores = {n: [] for n in n_trees}
    
    def process_single_run(run):
        current_seed = run
        
        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )

        run_ap_scores = {n: [] for n in n_trees}
        run_auc_scores = {n: [] for n in n_trees}

        for n in n_trees:
            rf = RandomForestClassifier(
                n_estimators=n, 
                random_state=current_seed, 
                min_samples_leaf=1, 
                max_features='sqrt'
            ).fit(val_data, val_labels)
            
            # Scoring full forests only
            y_pred_proba = rf.predict_proba(test_data)[:, 1]
            ap_scores = measure(test_labels, y_pred_proba)
            auc_scores = metrics.roc_auc_score(test_labels, y_pred_proba)

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
    
    np.savez(
        os.path.join(save_dir, f"{dataset_name}_rf_scores.npz"),
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
    main_save_dir = "results_rf"
    
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)
    
    for dataset_name in tqdm(odds_datasets.datasets_names, desc="Processing datasets"):
        data, labels = odds_datasets.load(dataset_name)
        
        for val_size in val_sizes:
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                compute_and_save_rf_scores(
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