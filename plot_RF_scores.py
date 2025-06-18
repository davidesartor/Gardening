import os
import odds_datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    plt.ylim(0, 1)
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
        plt.ylim(0, 1)
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
    
    # Set to 'by_val_size' for original plotting, 'by_n_trees' for new plotting
    PLOTTING_MODE = "by_n_trees"  # 'by_val_size' or 'by_n_trees'

    for dataset_name in tqdm(odds_datasets.datasets_names, desc="Processing datasets for plotting"):

        if PLOTTING_MODE == 'by_val_size':
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