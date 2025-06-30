import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import odds_datasets

def load_ap_scores(model_dir, dataset_name, val_size):
    val_dir = os.path.join(model_dir, f"val_size_{val_size}", "data")
    
    npz_path = os.path.join(val_dir, f"{dataset_name}_if_scores.npz")
    if os.path.exists(npz_path):
        return np.load(npz_path, allow_pickle=True)['ap_scores'].item()
    
    print(f"File '{npz_path}' not found.")
    return None

def plot_combined_results(ap_scores_if, ap_scores_gif, dataset_name, val_size, save_dir, n_trees, n_runs):
    plt.figure(figsize=(10, 7))
    plt.title(f"AP Comparison: IF vs GIF on {dataset_name} (Val Size: {val_size})")
    
    colors = plt.cm.tab10.colors[:len(n_trees)]
    linestyles = {'IF': '-', 'GIF': '--'}
    
    for idx, n in enumerate(n_trees):
        color = colors[idx]
        
        if ap_scores_if and n in ap_scores_if:
            if_scores = ap_scores_if[n]
            mean_if = np.mean(if_scores, axis=0)
            std_if = np.std(if_scores, axis=0)
            # Calculate Standard Error of the Mean
            sem_if = std_if / np.sqrt(n_runs)
            x_if = range(1, len(mean_if) + 1)
            plt.plot(x_if, mean_if, color=color, linestyle=linestyles['IF'], label=f'IF ({n} Trees)')
            # Use SEM for the error band
            plt.fill_between(x_if, mean_if - sem_if, mean_if + sem_if, color=color, alpha=0.1)
        else:
            print(f"Warning: IF scores for n={n} not found for {dataset_name}, val_size={val_size}")

        if ap_scores_gif and n in ap_scores_gif:
            gif_scores = ap_scores_gif[n]
            mean_gif = np.mean(gif_scores, axis=0)
            std_gif = np.std(gif_scores, axis=0)
            # Calculate Standard Error of the Mean
            sem_gif = std_gif / np.sqrt(gif_scores.shape[0])
            x_gif = range(1, len(mean_gif) + 1)
            plt.plot(x_gif, mean_gif, color=color, linestyle=linestyles['GIF'], label=f'GIF ({n} Trees)')
            # Use SEM for the error band
            plt.fill_between(x_gif, mean_gif - sem_gif, mean_gif + sem_gif, color=color, alpha=0.1)
        else:
            print(f"Warning: GIF scores for n={n} not found for {dataset_name}, val_size={val_size}")
    
    plt.xscale('log')
    plt.ylim(bottom=0.1)
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel(f'Average Precision Score (Avg ± SEM over {n_runs} runs)')
    plt.grid(True, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"ap_comparison_{dataset_name}_val_{val_size}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    n_runs = 50
    n_trees = [100, 300, 1000]
    val_sizes = [0.01, 0.05, 0.1, 0.2]
    models = {'IF': 'results_if', 'GIF': 'results_greedy_if'}
    
    for val_size in tqdm(val_sizes, desc="Overall Progress"):
        for dataset in tqdm(odds_datasets.datasets_names, desc=f"Val Size {val_size}", leave=False):
            
            ap_scores_if = load_ap_scores(models['IF'], dataset, val_size)
            ap_scores_gif = load_ap_scores(models['GIF'], dataset, val_size)
            
            if ap_scores_if is None or ap_scores_gif is None:
                print(f"Skipping plot for {dataset} with val size {val_size}: missing data.")
                continue

            save_dir = os.path.join("combined_plots_greedy", f"val_size_{val_size}")
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                plot_combined_results(ap_scores_if, ap_scores_gif, dataset, val_size, save_dir, n_trees, n_runs)
            except Exception as e:
                print(f"Error plotting results for {dataset} with val size {val_size}: {e}")
                continue