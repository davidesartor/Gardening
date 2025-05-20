from utils import split_data, score_growing_trees_rf

import os
import odds_datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def compute_and_save_rf_scores(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):
    
    all_ap_scores = {n: [] for n in n_trees}
    all_auc_scores = {n: [] for n in n_trees}
    
    for run in tqdm(range(n_runs), desc=f"{dataset_name} - RF Progress",leave=False):
        current_seed = run
        
        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )

        for n in n_trees:
            rf = RandomForestClassifier(n_estimators=n, random_state=current_seed, min_samples_leaf=1, max_features='sqrt',).fit(train_data, train_labels)
            
            ap_scores, auc_scores = score_growing_trees_rf(rf, val_data, val_labels, test_data, test_labels)
            
            all_ap_scores[n].append(ap_scores)
            all_auc_scores[n].append(auc_scores)

    ap_scores_dict = {n: np.array(all_ap_scores[n]) for n in n_trees}
    auc_scores_dict = {n: np.array(all_auc_scores[n]) for n in n_trees}
    
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(
        os.path.join(save_dir, f"{dataset_name}_rf_scores.npz"),
        ap_scores=ap_scores_dict,
        auc_scores=auc_scores_dict,
        allow_pickle=True
    )

def plot_avg_prc(ap_scores, color=None, label=None, verbose=False):
    
    mean_ap = np.mean(ap_scores, axis=0)
    std_ap = np.std(ap_scores, axis=0)
    x = range(1, len(mean_ap) + 1)
    
    if color is not None:
        plt.plot(x, mean_ap, color=color, label=label)
        plt.fill_between(x, mean_ap - std_ap, mean_ap + std_ap, color=color, alpha=0.2)
    else:
        plt.plot(x, mean_ap, label=label)
        plt.fill_between(x, mean_ap - std_ap, mean_ap + std_ap, alpha=0.2)
    if verbose:
        print("\n--- Average Precision Scores ---")
        print(f"Max Avg AP: {max(mean_ap):.4f} at {np.argmax(mean_ap) + 1} trees")

def plot_from_saved_data(save_dir, dataset_name, n_trees):
    
    data_rf = np.load(os.path.join(save_dir, f"{dataset_name}_rf_scores.npz"), allow_pickle=True)
    ap_scores_rf = data_rf['ap_scores'].item()

    original_save_dir = "results\\val_size_0.2\\data"

    data_if = np.load(os.path.join(original_save_dir, f"{dataset_name}_scores.npz"), allow_pickle=True)
    ap_scores_if = data_if['ap_scores'].item()
    
    plt.figure(figsize=(10, 7))
    plt.title(f"Average Precision Score vs Number of Trees on {dataset_name}")
    
    colormap = plt.colormaps["tab20"]
    
    for i, n in enumerate(n_trees):
        ap_scores = ap_scores_if[n]
        plot_avg_prc(ap_scores, color=colormap.colors[2*i], label=f'IF {n} Trees')
        ap_scores = ap_scores_rf[n]
        plot_avg_prc(ap_scores, color=colormap.colors[2*i+1], label=f'RF {n} Trees')
   
    plt.xscale('log')
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('Average Precision Score (Avg +/- Std Dev over 10 runs)')
    plt.grid(True, which="both")
    plt.legend()
    
    save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, "plots")
    
    plt.savefig(os.path.join(save_dir, f"comparison_{dataset_name}.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
   
    n_runs = 10
    n_trees = [100]
    val_sizes = [0.2]
    test_size = 0.2
    main_save_dir = "results_rf"
   
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)
    
    for dataset_name in tqdm(odds_datasets.small_datasets_names, desc="Processing datasets"):
        data, labels = odds_datasets.load(dataset_name)
        
        for val_size in val_sizes:
           
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            plots_dir = os.path.join(val_size_dir, "plots")
            
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
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
            
            plot_from_saved_data(save_dir=data_dir, dataset_name=dataset_name, n_trees=n_trees)

