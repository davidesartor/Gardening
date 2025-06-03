from utils import split_data, measure

import os
import odds_datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def plot_all_val_sizes(dataset_name, val_sizes, n_trees, main_save_dir, n_runs):
    plt.figure(figsize=(10, 7))
    colormap = plt.colormaps["tab10"]
    
    for idx, val_size in enumerate(val_sizes):
        data_path = os.path.join(main_save_dir, f"val_size_{val_size}", "data", f"{dataset_name}_rf_scores.npz")
        if not os.path.exists(data_path):
            print(f"Data for {dataset_name} with val_size {val_size} not found, skipping.")
            continue
        data = np.load(data_path, allow_pickle=True)
        ap_scores_dict = data['ap_scores'].item()
        
        means = []
        stds = []
        for n in n_trees:
            ap_scores = ap_scores_dict[n]  # Shape: (n_runs,)
            mean_ap = np.mean(ap_scores)
            std_ap = np.std(ap_scores)
            means.append(mean_ap)
            stds.append(std_ap)
        
        color = colormap(idx)
        plt.errorbar(n_trees, means, yerr=stds, fmt='o-', capsize=5, color=color, label=f'Val Size {val_size}')
    
    plt.title(f"Average Precision Score vs Number of Trees on {dataset_name}")
    plt.xlabel('Number of Trees in Forest')
    plt.ylabel(f'Average Precision Score (Avg +/- Std Dev over {n_runs} runs)')
    plt.legend()
    plt.grid(True)
    
    plots_dir = os.path.join(main_save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"avg_precision_{dataset_name}.pdf"), bbox_inches='tight')
    plt.close()

def plot_by_n_trees_rf(main_save_dir, dataset_name, n_trees_list, val_sizes_list, n_runs):
    
    aggregated_ap_data = {n: {} for n in n_trees_list}

    for val_size in val_sizes_list:
        data_dir = os.path.join(main_save_dir, f"val_size_{val_size}", "data")
        file_path = os.path.join(data_dir, f"{dataset_name}_rf_scores.npz")

        if not os.path.exists(file_path):
            print(f"Warning: Data file not found for {dataset_name}, val_size={val_size}. Skipping.")
            continue

        loaded_data = np.load(file_path, allow_pickle=True)
        ap_scores_dict_for_val_size = loaded_data['ap_scores'].item()

        for n_estimators_key in n_trees_list:
            if n_estimators_key in ap_scores_dict_for_val_size:
                aggregated_ap_data[n_estimators_key][val_size] = ap_scores_dict_for_val_size[n_estimators_key]
            else:
                print(f"Warning: No data for n_trees={n_estimators_key} in {file_path}. Skipping.")

    for n_trees_val in n_trees_list:
        plot_save_dir = os.path.join(main_save_dir, f"plots_n_trees_{n_trees_val}")
        os.makedirs(plot_save_dir, exist_ok=True)

        plt.figure(figsize=(10, 7))
        plt.title(f"Average Precision (N={n_trees_val}) on {dataset_name}")

        colormap = plt.colormaps["tab10"]
        colors = [colormap(i) for i in range(len(val_sizes_list))]

        means_across_val_sizes = []
        stds_across_val_sizes = []
        valid_val_sizes_for_plot = []

        for val_size in val_sizes_list:
            if val_size in aggregated_ap_data[n_trees_val]:
                ap_scores = aggregated_ap_data[n_trees_val][val_size]
                means_across_val_sizes.append(np.mean(ap_scores))
                stds_across_val_sizes.append(np.std(ap_scores))
                valid_val_sizes_for_plot.append(str(val_size)) # Convert to string for x-axis ticks

        plt.errorbar(valid_val_sizes_for_plot, means_across_val_sizes, yerr=stds_across_val_sizes,
                     fmt='o-', capsize=5, color=colors[0]) # Use first color, as lines are for N_trees

        plt.xlabel('Validation Set Size')
        plt.ylabel(f'Average Precision (Avg +/- Std Dev over {n_runs} runs)')
        plt.grid(True)
        plt.savefig(os.path.join(plot_save_dir, f"avg_precision_{dataset_name}_N_{n_trees_val}.pdf"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Parameters
    n_runs = 10
    n_trees = [100, 300, 1000]
    val_sizes = [0.01, 0.05, 0.1, 0.2] 
    test_size = 0.2
    main_save_dir = "results_rf"
    
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)
    
    # Set to True to compute and save scores, False to only plot from existing data
    COMPUTE_AND_SAVE_DATA = False
    # Set to 'by_val_size' for original plotting, 'by_n_trees' for new plotting
    PLOTTING_MODE = "by_val_size"  # 'by_val_size' or 'by_n_trees' or None

    for dataset_name in tqdm(odds_datasets.datasets_names, desc="Processing datasets"):
        data, labels = odds_datasets.load(dataset_name)
        
        if COMPUTE_AND_SAVE_DATA:
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
        
        if PLOTTING_MODE is None:
            continue

        elif PLOTTING_MODE == 'by_val_size':

            # This call happens once per dataset, after all val_sizes for that dataset have been processed.
            plot_all_val_sizes(dataset_name, val_sizes, n_trees, main_save_dir, n_runs)
        
        elif PLOTTING_MODE == 'by_n_trees':
            # New plotting mode: one plot per n_trees, showing val_sizes on x-axis
            plot_by_n_trees_rf(
                main_save_dir=main_save_dir,
                dataset_name=dataset_name,
                n_trees_list=n_trees,
                val_sizes_list=val_sizes,
                n_runs=n_runs
            )