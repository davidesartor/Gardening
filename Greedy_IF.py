import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
from matplotlib import pyplot as plt

from utils import compute_tree_anomaly_scores, measure, split_data

import os
import odds_datasets
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed


def score_growing_trees(sk_IF, val_data, val_labels, test_data, test_labels):
    avg_precision_scores = []
    auc_scores = []

    n_trees = len(sk_IF.estimators_)

    # Get tree indices using greedy approach
    ordered_indices = sorted_indices_trees_greedy(sk_IF, val_data, val_labels)
    tree_train = compute_tree_anomaly_scores(sk_IF, test_data)  # Shape: (n_trees, n_test_samples)
    tree_train_ordered = tree_train[ordered_indices, :]  # Reorder by greedy selection

    # Compute cumulative geometric mean of anomaly scores
    scores = np.exp(np.cumsum(np.log(tree_train_ordered), axis=0).T / np.arange(1, n_trees+1))
    scores = scores.T  # Shape: (n_trees, n_test_samples)

    # Evaluate performance for each number of trees
    for i in range(n_trees):
        y_pred = scores[i]  # Scores using first i+1 trees
        avg_precision_scores.append(measure(test_labels, y_pred))
        auc_scores.append(metrics.roc_auc_score(test_labels, y_pred))

    return avg_precision_scores, auc_scores

def sorted_indices_trees_greedy(sk_IF, val_data, val_labels):
    n_trees = len(sk_IF.estimators_)
    tree_scores = compute_tree_anomaly_scores(sk_IF, val_data)
    
    # Find best initial tree
    ap_scores = Parallel(n_jobs=2)(delayed(measure)(val_labels, tree) for tree in tree_scores)
    best_tree_idx = np.argmax(ap_scores)
    selected_indices = [best_tree_idx]
    available_indices = np.ones(n_trees, dtype=bool)
    available_indices[best_tree_idx] = False
    cumulative_log_scores = np.log(tree_scores[best_tree_idx])

    # Greedy selection over available candidates
    for _ in tqdm(range(1, n_trees), leave=False):
        candidate_indices = np.where(available_indices)[0]
        if len(candidate_indices) == 0:
            break
        
        # Compute combined scores only for candidates
        temp_combined = np.exp(
            (cumulative_log_scores + np.log(tree_scores[candidate_indices])) / 
            (len(selected_indices) + 1)
        )
        
        ap_scores = Parallel(n_jobs=-1)(delayed(measure)(val_labels, score) for score in temp_combined)
        best_candidate_idx = np.argmax(ap_scores)
        best_idx = candidate_indices[best_candidate_idx]
        
        selected_indices.append(best_idx)
        available_indices[best_idx] = False
        cumulative_log_scores += np.log(tree_scores[best_idx])
        
    return np.array(selected_indices)

def compute_and_save_scores(dataset_name, data, labels, n_trees, n_runs, val_size, test_size, save_dir):

    # dictionaries to store scores
    all_ap_scores = {n: [] for n in n_trees}
    all_auc_scores = {n: [] for n in n_trees}

    def process_single_run(run):
        current_seed = run

        train_data, _, val_data, val_labels, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=current_seed
        )

        run_ap_scores = {n: [] for n in n_trees}
        run_auc_scores = {n: [] for n in n_trees}

        # computing scores for each number of trees
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

    # converting lists of lists to 2D NumPy arrays (n_runs, n) 
    ap_scores_dict = {n: np.array(all_ap_scores[n]) for n in n_trees}
    auc_scores_dict = {n: np.array(all_auc_scores[n]) for n in n_trees}

    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, f"{dataset_name}_if_scores.npz"),
        ap_scores=ap_scores_dict,
        auc_scores=auc_scores_dict,
        allow_pickle=True
    )


def compute_bruteforce_points(dataset_name, data, labels, n_trees, n_runs, bf_values, max_iter, val_size, test_size,
                              save_dir):
    all_ap_bf_scores = {n: [] for n in n_trees}
    all_max_ap_bf_scores = {n: [] for n in n_trees}

    def process_single_bf_run(run_seed, n_estimators):

        np.random.seed(run_seed)

        train_data, _, val_data, val_label, test_data, test_labels = split_data(
            data, labels, val_size=val_size, test_size=test_size, random_state=run_seed
        )

        sk_IF = IsolationForest(n_estimators=n_estimators, random_state=run_seed, n_jobs=-1).fit(train_data)
        tree_train = compute_tree_anomaly_scores(sk_IF, val_data).astype(np.float32)  # (n_estimators, n_test)
        tree_test = compute_tree_anomaly_scores(sk_IF, test_data).astype(np.float32)

        ap_bf_scores_for_run = []
        max_ap_bf_scores_for_run = []

        # Precompute random indices
        random_indices = {
            x: [np.random.choice(n_estimators, x, replace=False) for _ in range(max_iter)]
            for x in bf_values if x <= n_estimators
        }

        for x in bf_values:
            if x <= n_estimators:
                # Use list comprehension
                scores_train = [np.exp(np.mean(np.log(tree_train[idx, :]), axis=0)) for idx in random_indices[x]]
                scores_test = [np.exp(np.mean(np.log(tree_test[idx, :]), axis=0)) for idx in random_indices[x]]
                ap_train_list_for_x = [measure(val_label, s) for s in scores_train]
                ap_test_list_for_x = [measure(val_label, s) for s in scores_test]
                
                max_index = np.argmax(np.array(ap_train_list_for_x))

                ap_bf_scores_for_run.append(np.mean(ap_test_list_for_x))
                max_ap_bf_scores_for_run.append(ap_test_list_for_x[max_index])
            else:
                ap_bf_scores_for_run.append(np.nan)
                max_ap_bf_scores_for_run.append(np.nan)

        return ap_bf_scores_for_run, max_ap_bf_scores_for_run

    # Top-level parallelism over (run, n)
    results = Parallel(n_jobs=-1)(
        delayed(process_single_bf_run)(run, n)
        for run in tqdm(range(n_runs), desc=f"{dataset_name} - Brute-Force Runs", leave=False)
        for n in n_trees
    )

    # Reshape results into dictionary format
    result_idx = 0
    for _ in range(n_runs):
        for n in n_trees:
            ap_scores, max_ap_scores = results[result_idx]
            all_ap_bf_scores[n].append(ap_scores)
            all_max_ap_bf_scores[n].append(max_ap_scores)
            result_idx += 1

    # Convert to numpy arrays
    ap_bf_scores_dict = {n: np.array(all_ap_bf_scores[n]) for n in n_trees}
    max_ap_bf_scores_dict = {n: np.array(all_max_ap_bf_scores[n]) for n in n_trees}

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"{dataset_name}_if_bf_scores.npz"),
        ap_scores=ap_bf_scores_dict,
        max_ap_scores=max_ap_bf_scores_dict,
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

# Parameters
n_runs = 10
n_trees = [100, 300, 1000]
val_sizes = [0.01, 0.05, 0.1, 0.2]
test_size = 0.2
max_iter = 5000
bf_values = [10, 50, 100, 300, 800] # Increased bf_values to include points for higher n_trees
main_save_dir = "results_greedy_if"

# creating main results directory if it doesn't exist
if not os.path.exists(main_save_dir):
    os.makedirs(main_save_dir)


for dataset_name in tqdm(odds_datasets.small_datasets_names, desc="Processing datasets"):

    data, labels = odds_datasets.load(dataset_name)

    for val_size in val_sizes:

        # creating subdirectories for each validation size
        val_size_dir = os.path.join(main_save_dir, f"val_size_{val_size}")
        data_dir = os.path.join(val_size_dir, "data")
        plots_dir = os.path.join(val_size_dir, "plots")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # computing and saving greedy scores
        try:
            pass
            #compute_and_save_scores(
            #    dataset_name=dataset_name,
            #    data=data,
            #    labels=labels,
            #    n_trees=n_trees,
            #    n_runs=n_runs,
            #    val_size=val_size,
            #    test_size=test_size,
            #    save_dir=data_dir
            #)

        except ValueError as e:
            print(f"Error processing {dataset_name} with val_size {val_size}: {e}")
            continue

        # generating plots from saved greedy data
        #plot_from_saved_data(save_dir=data_dir, dataset_name=dataset_name, n_trees=n_trees, n_runs=n_runs)

        # computing bruteforce points
        try:
            #compute_bruteforce_points(
            #    dataset_name=dataset_name,
            #    data=data,
            #    labels=labels,
            #    n_trees=n_trees,
            #    n_runs=n_runs,
            #    bf_values=bf_values,
            #    max_iter=max_iter,
            #    val_size=val_size,
            #    test_size=test_size,
            #    save_dir=data_dir
            #)

            # plotting bruteforce points
            plot_bf_points(
                save_dir=data_dir,
                dataset_name=dataset_name,
                n_trees=n_trees,
                bf_values=bf_values,
                n_runs=n_runs,
                max_iter=max_iter
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {dataset_name} with val_size {val_size} for bruteforce: {e}")
            continue
