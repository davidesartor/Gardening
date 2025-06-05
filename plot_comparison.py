import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import odds_datasets

def load_ap_scores(model_dir, dataset_name, val_size):
    val_dir = os.path.join(model_dir, f"val_size_{val_size}", "data")
    
    # loading Isolation Forest scores
    npz_path_if = os.path.join(val_dir, f"{dataset_name}_if_scores.npz")
    if os.path.exists(npz_path_if):
        return np.load(npz_path_if, allow_pickle=True)['ap_scores'].item()
    
    # If IF scores not found, loading Random Forest scores
    npz_path_rf = os.path.join(val_dir, f"{dataset_name}_rf_scores.npz")
    if os.path.exists(npz_path_rf):
        return np.load(npz_path_rf, allow_pickle=True)['ap_scores'].item()
    
    # If neither is found
    print(f"Neither '{npz_path_if}' nor '{npz_path_rf}' found.")
    return None

def plot_combined_results(ap_scores_if, ap_scores_rf, dataset_name, val_size, save_dir, n_trees):
    plt.figure(figsize=(10, 7))
    plt.title(f"AP Comparison: IF vs RF on {dataset_name} (Val Size: {val_size})")
    
    colors = plt.cm.tab10.colors[:len(n_trees)]
    linestyles = {'IF': '-', 'RF': '--'}  # RF dashed line

    # ---- Plot IF results ----
    for idx, n in enumerate(n_trees):
        color = colors[idx]
        if_scores = ap_scores_if[n]
        mean_if = np.mean(if_scores, axis=0)
        std_if = np.std(if_scores, axis=0)
        x = range(1, len(mean_if) + 1)
        
        plt.plot(x, mean_if, color=color, linestyle=linestyles['IF'], label=f'IF ({n} Trees)')
        plt.fill_between(x, mean_if - std_if, mean_if + std_if, color=color, alpha=0.1)

    # ---- Plot single RF line ----
    all_rf_means = [np.mean(ap_scores_rf[n]) for n in n_trees]
    rf_avg = np.mean(all_rf_means)

    x_vals = range(1, len(mean_if) + 1)
    plt.plot(x_vals, [rf_avg] * len(x_vals), linestyle=linestyles['RF'], color='black', label='RF (Avg over Tree Counts)')

    # ---- Final formatting ----
    plt.xscale('log')
    plt.ylim(bottom=0.1)
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('Average Precision Score (Avg ± Std over 10 runs)')
    plt.grid(True, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"ap_comparison_{dataset_name}_val_{val_size}.pdf")
    plt.ylim(0, 1)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()





if __name__ == "__main__":

    n_trees = [100, 300, 1000]
    val_sizes = [0.01, 0.05, 0.1, 0.2]
    models = {'IF': 'results_if', 'RF': 'results_rf'}  # model directories
    
    for val_size in tqdm(val_sizes, desc="Overall Progress"):
        for dataset in tqdm(odds_datasets.datasets_names, desc=f"Val Size {val_size}", leave=False):
           
            # loading scores for both models
            ap_scores_if = load_ap_scores(models['IF'], dataset, val_size)
            ap_scores_rf = load_ap_scores(models['RF'], dataset, val_size)
            
            save_dir = os.path.join("combined_plots", f"val_size_{val_size}")
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                # generate and save the combined plot
                plot_combined_results(ap_scores_if, ap_scores_rf, dataset, val_size, save_dir, n_trees)

            except TypeError as e:
                print(f"Error plotting results for {dataset} with val size {val_size}: no file found")
                continue