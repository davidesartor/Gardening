import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
from matplotlib import pyplot as plt

from utils import split_data, score_growing_trees

import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed


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

    data = np.load(os.path.join(save_dir, f"{dataset_name}_if_scores.npz"), allow_pickle=True)
    # Retrieve the dictionary of AP scores
    ap_scores_dict = data['ap_scores'].item()

    # Set up the plot
    plt.figure(figsize=(10, 7))
    plt.title(f"Average Precision Score vs Number of Trees on {dataset_name}")

    colormap = plt.colormaps["tab10"]
    # Plot for each number of trees
    for n, color in zip(n_trees, colormap.colors):
        ap_scores = ap_scores_dict[n]  # Shape: (n_runs, n)
        plot_avg_prc(ap_scores, color=color, label=f'{n} Trees')

    # Configure axes
    plt.xscale('log')
    plt.xlabel('Number of Trees Used (Cumulative, Log Scale)')
    plt.ylabel(f'Average Precision Score (Avg +/- Std Dev over {n_runs} runs)')
    plt.grid(True, which="both")
    plt.legend()
    save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, "plots")
    plt.savefig(os.path.join(save_dir, f"avg_precision_{dataset_name}.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # parameters
    n_runs = 10
    n_trees = [100, 300, 1000] # Max 10 forests (using tab10 colormap)
    val_sizes = [0.01, 0.05, 0.1, 0.2] 
    test_size = 0.2
    main_save_dir = "results_if"


    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    # Iterate over datasets
    for dataset_name in tqdm(odds_datasets.small_datasets_names, desc="Processing datasets"):
        # Load dataset
        data, labels = odds_datasets.load(dataset_name)

        # Iterate over validation sizes
        for val_size in val_sizes:

            # Create a subdirectories
            val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
            data_dir = os.path.join(val_size_dir, "data")
            plots_dir = os.path.join(val_size_dir, "plots")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            try:
                # Compute and save scores
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

            # Generate and save plots
            plot_from_saved_data(save_dir=data_dir, dataset_name=dataset_name, n_trees=n_trees)