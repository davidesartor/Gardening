import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
from matplotlib import pyplot as plt
import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from utils import plot_avg_prc


def plot_from_saved_data(save_dir, dataset_name, n_trees, n_runs):

    data = np.load(os.path.join(save_dir, f"{dataset_name}_if_scores.npz"), allow_pickle=True)
    # retrieving the data
    ap_scores_dict = data['ap_scores'].item()

    plt.figure(figsize=(10, 7))
    plt.title(f"Average Precision Score vs Number of Trees on {dataset_name}")

    colormap = plt.colormaps["tab10"]
    # plotting for each number of trees
    for n, color in zip(n_trees, colormap.colors):
        ap_scores = ap_scores_dict[n]  # (n_runs, n)
        plot_avg_prc(ap_scores, color=color, label=f'{n} Trees')

    # configuring axes
    plt.xscale('log')
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel(f'Average Precision Score (Avg +/- Std Dev over {n_runs} runs)')
    plt.grid(True, which="both")
    plt.legend()
    save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, "plots")
    plt.savefig(os.path.join(save_dir, f"avg_precision_{dataset_name}.pdf"), bbox_inches='tight')
    plt.close()

def plot_bf_points(save_dir, dataset_name, n_trees, bf_values, n_runs, max_iter):
    # Load greedy scores
    data = np.load(os.path.join(save_dir, f"{dataset_name}_if_scores.npz"), allow_pickle=True)
    ap_scores_dict = data['ap_scores'].item()

    # Load brute-force scores
    bf_data = np.load(os.path.join(save_dir, f"{dataset_name}_if_bf_scores.npz"), allow_pickle=True)
    ap_bf_scores_dict = bf_data['ap_scores'].item()
    max_ap_bf_scores_dict = bf_data['max_ap_scores'].item()

    plt.figure(figsize=(10, 7))
    plt.title(f"Average Precision Score vs Number of Trees on {dataset_name}")

    colormap = plt.colormaps["tab10"]
    
    # Iterate through each n in n_trees to plot corresponding greedy and brute-force data
    for n_idx, (n, color) in enumerate(zip(n_trees, colormap.colors)):
        # Plot greedy curves
        ap_scores = ap_scores_dict[n]  # (n_runs, n)
        plot_avg_prc(ap_scores, color=color, label=f'Greedy ({n} Trees)')

        # Plot brute-force points for the current n
        if n in ap_bf_scores_dict:
            ap_bf_scores_for_n = ap_bf_scores_dict[n]
            max_ap_bf_scores_for_n = max_ap_bf_scores_dict[n]

            # Filter bf_values to include only x <= current n
            valid_bf_values_indices = [idx for idx, x_val in enumerate(bf_values) if x_val <= n]
            
            if len(valid_bf_values_indices) > 0:
                current_bf_x_vals = np.array(bf_values)[valid_bf_values_indices]

                # Plot mean brute-force AP with error bars
                # Ensure we only take valid (non-NaN) data for mean/std calculation from the correct slice
                mean_ap_bf = np.array([np.mean(ap_bf_scores_for_n[:, i][~np.isnan(ap_bf_scores_for_n[:, i])]) 
                                       for i in valid_bf_values_indices])
                std_ap_bf = np.array([np.std(ap_bf_scores_for_n[:, i][~np.isnan(ap_bf_scores_for_n[:, i])]) 
                                      for i in valid_bf_values_indices])
                
                # Check for valid mean values before plotting
                valid_mean_ap_indices = ~np.isnan(mean_ap_bf)
                if np.any(valid_mean_ap_indices):
                    plt.errorbar(current_bf_x_vals[valid_mean_ap_indices], mean_ap_bf[valid_mean_ap_indices], 
                                 yerr=std_ap_bf[valid_mean_ap_indices], fmt='o', color=color, capsize=5, 
                                 label=f'BF Avg ({n} Trees, {max_iter} iter)', # Unique label for each n
                                 markerfacecolor=color, markeredgecolor='black', markersize=6)

                # Plot the highest AP point obtained in the brute-force search
                highest_overall_ap = np.array([np.max(max_ap_bf_scores_for_n[:, i][~np.isnan(max_ap_bf_scores_for_n[:, i])]) 
                                               if np.any(~np.isnan(max_ap_bf_scores_for_n[:, i])) else np.nan
                                               for i in valid_bf_values_indices])
                
                # Check for valid max values before plotting
                valid_highest_ap_indices = ~np.isnan(highest_overall_ap)
                if np.any(valid_highest_ap_indices):
                    plt.plot(current_bf_x_vals[valid_highest_ap_indices], highest_overall_ap[valid_highest_ap_indices], 
                             'X', color=color, markersize=8, 
                             label=f'BF Highest ({n} Trees)', # Unique label for each n
                             markeredgecolor='black')
                             
    # Configuring axes
    plt.xscale('log')
    plt.xlabel('Number of Trees Used (Cumulative for Greedy, Specified for Brute-Force, Log Scale)')
    plt.ylabel(f'Average Precision Score (Avg +/- Std Dev over {n_runs} runs)')
    plt.grid(True, which="both")
    plt.legend()
    save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, "plots")
    plt.savefig(os.path.join(save_dir, f"avg_precision_{dataset_name}_with_bf.pdf"), bbox_inches='tight')
    plt.close()

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

    for dataset_name in tqdm(odds_datasets.small_datasets_names, desc="Plotting for datasets"):
        for val_size in val_sizes:
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            plots_dir = os.path.join(val_size_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            try:
                print(f"Plotting results for {dataset_name}, val_size={val_size}...")
                plot_bf_points(
                    save_dir=data_dir,
                    dataset_name=dataset_name,
                    n_trees=n_trees,
                    bf_values=bf_values,
                    n_runs=n_runs,
                    max_iter=max_iter
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"Error plotting {dataset_name} with val_size {val_size}: {e}. Data files might be missing.")
                continue