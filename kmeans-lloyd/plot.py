import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_kmeans_results(A: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
    num_dimensions, k = A.shape[1], centroids.shape[0]
    title_suffix = ""
    xlabel, ylabel = "Dimension 1", "Dimension 2"

    if num_dimensions > 2:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(A)
        centroids_plot = pca.transform(centroids)
        xlabel, ylabel = "Principal Component 1", "Principal Component 2"
        title_suffix = " (PCA Projection)"
    elif num_dimensions == 2:
        X_plot, centroids_plot = A, centroids
    else:
        print(f"Plotting requires >= 2 dimensions, got {num_dimensions}.")
        return

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis', k)

    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap=cmap, marker='o', alpha=0.7, label='Data Points')
    plt.scatter(centroids_plot[:, 0], centroids_plot[:, 1], c='red', marker='X', s=200, label='Final Centroids')

    plt.title(f'K-Means Clustering (k={k}){title_suffix}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()