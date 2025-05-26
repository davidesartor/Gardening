import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import odds_datasets

def load_ap_scores(model_dir, dataset_name, val_size):

    val_dir = os.path.join(model_dir, f"val_size_{val_size}", "data")
    try:
        npz_path = os.path.join(val_dir, f"{dataset_name}_if_scores.npz")
        data = np.load(npz_path, allow_pickle=True)

    except:
        npz_path = os.path.join(val_dir, f"{dataset_name}_rf_scores.npz")
        data = np.load(npz_path, allow_pickle=True)

   
    return data['ap_scores'].item()

def plot_combined_results(ap_scores_if, ap_scores_rf, dataset_name, val_size, save_dir, n_trees):
   
    plt.figure(figsize=(10, 7))
    plt.title(f"AP Comparison: IF vs RF on {dataset_name} (Val Size: {val_size})")
    
    colors = plt.cm.tab10.colors[:len(n_trees)]  # using distinct colors for each tree count
    linestyles = {'IF': '-', 'RF': '--'}  
    
    for idx, n in enumerate(n_trees):
        color = colors[idx]
        
        # Plotting Isolation Forest results
        if_scores = ap_scores_if[n]
        mean_if = np.mean(if_scores, axis=0)
        std_if = np.std(if_scores, axis=0)
        x = range(1, len(mean_if) + 1)
        plt.plot(x, mean_if, color=color, linestyle=linestyles['IF'], label=f'IF ({n} Trees)')
        plt.fill_between(x, mean_if - std_if, mean_if + std_if, color=color, alpha=0.1)
        
        # Plotting Random Forest results (growing, unused for now)
        rf_scores = ap_scores_rf[n]
        mean_rf = np.mean(rf_scores, axis=0)
        std_rf = np.std(rf_scores, axis=0)

        mean_rf = np.array([mean_rf ]*x[-1])
        std_rf = np.array([std_rf ]*x[-1])   


        # plt.plot(x, mean_rf, color=color, linestyle=linestyles['RF'], label=f'RF ({n} Trees)')
        # plt.fill_between(x, mean_rf - std_rf, mean_rf + std_rf, color=color, alpha=0.1)

        plt.plot(x, mean_rf, color=color, linestyle=linestyles['RF'], label=f'RF ({n} Trees)')
        plt.errorbar(idx+1, mean_rf[0], yerr=std_rf[0], fmt='o-', capsize=5, color=color)


    
    plt.xscale('log')
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel('Average Precision Score (Avg ± Std over 10 runs)')
    plt.grid(True, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # saving the plot
    save_path = os.path.join(save_dir, f"ap_comparison_{dataset_name}_val_{val_size}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    n_trees = [100, 300, 1000]
    val_sizes = [ 0.05, 0.1, 0.2] 
    models = {'IF': 'results_if', 'RF': 'results_rf'}  # model directories
    
    for val_size in tqdm(val_sizes, desc="Overall Progress"):
        for dataset in tqdm(odds_datasets.small_datasets_names, desc=f"Val Size {val_size}", leave=False):
           
            # loading scores for both models
            ap_scores_if = load_ap_scores(models['IF'], dataset, val_size)
            ap_scores_rf = load_ap_scores(models['RF'], dataset, val_size)
            
            save_dir = os.path.join("combined_plots", f"val_size_{val_size}")
            os.makedirs(save_dir, exist_ok=True)
            
            # generate and save the combined plot
            plot_combined_results(ap_scores_if, ap_scores_rf, dataset, val_size, save_dir, n_trees)
