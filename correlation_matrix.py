import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from utils import compute_tree_anomaly_scores, measure, split_data
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
from matplotlib import pyplot as plt
import seaborn as sns

# Control variable to switch between compute+plot and plot-only modes
PLOT_ONLY = False

def score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels):
    tree_train = compute_tree_anomaly_scores(sk_IF, test_data)
    ap_scores = Parallel(n_jobs=-1)(
        delayed(measure)(test_labels, tree_scores) for tree_scores in tree_train
    )
    return np.array(ap_scores)

def compute_and_save_correlation(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):
    all_ap_scores = {n: [] for n in n_trees}

    def process_single_run(run):
        current_seed = run
        train_data, _, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )
        run_ap_scores = {n: [] for n in n_trees}
        for n in n_trees:
            sk_IF = IsolationForest(n_estimators=n, random_state=current_seed).fit(train_data)
            ap_scores = score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels)
            run_ap_scores[n] = ap_scores
        return run_ap_scores

    results = Parallel(n_jobs=10)(
        delayed(process_single_run)(run)
        for run in tqdm(range(n_runs), desc=f"{dataset_name} - Runs", leave=False)
    )

    for run_ap in results:
        for n in n_trees:
            all_ap_scores[n].append(run_ap[n])

    corr_matrices = {}
    for n in n_trees:
        ap_scores_array = np.array(all_ap_scores[n])
        if ap_scores_array.shape[1] > 1:
            corr_matrix = np.corrcoef(ap_scores_array, rowvar=False)
            corr_matrices[n] = corr_matrix
        else:
            corr_matrices[n] = np.array([[1.0]])

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"{dataset_name}_if_ap_corr.npz"),
        corr_matrices=corr_matrices,
        allow_pickle=True
    )
    return corr_matrices

def plot_correlation_matrices(save_dir, dataset_name, n_trees, max_matrix_size=200):
    
    plots_dir = os.path.join(os.path.dirname(save_dir), "plots")
    
    try:
        os.makedirs(plots_dir, exist_ok=True)
        if not os.access(plots_dir, os.W_OK):
            raise PermissionError(f"No write permissions for {plots_dir}")
        print(f"Plots directory: {plots_dir}")
    except Exception as e:
        print(f"Error creating plots directory {plots_dir}: {e}")
        return

    try:
        data = np.load(os.path.join(save_dir, f"{dataset_name}_if_ap_corr.npz"), allow_pickle=True)
        corr_matrices = data['corr_matrices'].item()
    except FileNotFoundError:
        print(f"Error: Correlation matrix file not found for {dataset_name} in {save_dir}")
        return
    except Exception as e:
        print(f"Error loading .npz file for {dataset_name}: {e}")
        return

    def plot_single_heatmap(n):
        if n not in corr_matrices:
            print(f"Warning: No correlation matrix found for {n} trees in {dataset_name}")
            return
        
        corr_matrix = corr_matrices[n]
        
        if corr_matrix.shape[0] > max_matrix_size:
            step = corr_matrix.shape[0] // max_matrix_size + 1
            indices = np.arange(0, corr_matrix.shape[0], step)
            corr_matrix = corr_matrix[indices][:, indices]
            #print(f"Downsampled {n}-tree matrix to {corr_matrix.shape[0]}x{corr_matrix.shape[0]}")
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                corr_matrix,
                annot=False,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'},
                xticklabels=False,
                yticklabels=False
            )
            plt.title(f"Correlation Matrix of AP Scores ({n} Trees) - {dataset_name}")
            plt.xlabel("")
            plt.ylabel("")
            output_path = os.path.join(plots_dir, f"corr_matrix_{dataset_name}_{n}_trees.pdf")
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close()

        except Exception as e:
            print(f"Error saving plot for {dataset_name}, {n} trees: {e}")

    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(plot_single_heatmap)(n)
        for n in tqdm(n_trees, desc=f"Plotting {dataset_name}", leave=False)
    )

# Parameters
n_runs = 10
n_trees = [100, 300, 1000]
val_sizes = [0.01, 0.05, 0.1, 0.2]
test_size = 0.2
main_save_dir = "results_ap_correlation"

os.makedirs(main_save_dir, exist_ok=True)

for dataset_name in tqdm(odds_datasets.datasets_names, desc="Processing datasets"):
    try:
        data, labels = odds_datasets.load(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        continue
    for val_size in val_sizes:
        val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
        data_dir = os.path.join(val_size_dir, "data")
        print(f"Processing {dataset_name} with val_size {val_size}, data_dir: {data_dir}")
        try:
            os.makedirs(data_dir, exist_ok=True)
            if PLOT_ONLY and not os.path.exists(os.path.join(data_dir, f"{dataset_name}_if_ap_corr.npz")):
                print(f"Skipping {dataset_name} with val_size {val_size}: .npz file missing")
                continue
            if not PLOT_ONLY:
                compute_and_save_correlation(
                    dataset_name=dataset_name,
                    data=data,
                    labels=labels,
                    n_trees=n_trees,
                    n_runs=n_runs,
                    val_size=val_size,
                    test_size=test_size,
                    save_dir=data_dir
                )
            plot_correlation_matrices(
                save_dir=data_dir,
                dataset_name=dataset_name,
                n_trees=n_trees,
                max_matrix_size=200
            )
        except Exception as e:
            print(f"Error processing {dataset_name} with val_size {val_size}: {e}")
            continue