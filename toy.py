import numpy as np
import matplotlib.pyplot as plt

# Generating two randomly distributed clusters of n points with n/20 anomalies
def double_cluster_with_generator(seed):
    rng = np.random.default_rng(seed)

    pts = 4000
    std = 0.1
    step = 0.4
    sx_cluster = rng.standard_normal((pts, 2)) * std + step
    dx_cluster = rng.standard_normal((pts, 2)) * std - step

    anomaly = (rng.random((round(pts / 20), 2)) - 0.5) * 2

    data = np.vstack([sx_cluster, dx_cluster, anomaly])
    labels = (np.linalg.norm(data + step, axis=1) > std * 3) & (np.linalg.norm(data - step, axis=1) > std * 3)

    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], c=1 - labels, cmap='Set1')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xticks([])
    plt.yticks([])

    return data, labels
