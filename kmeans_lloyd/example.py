import time
from sklearn.datasets import make_blobs
from kmeans import kmeans
from plot import plot_kmeans_results

if __name__ == "__main__":
    print("="*30)
    print(" K-Means Example with 2D Data ")
    print("="*30)
    k_clusters_2d = 4

    X_data_2d, y_true_2d = make_blobs(n_samples=300, centers=k_clusters_2d, 
                                      n_features=2, cluster_std=0.8, 
                                      random_state=43)

    print(f"\nGenerated 2D data with shape: {X_data_2d.shape}")
    print(f"Running K-Means for k={k_clusters_2d}...")

    start_time = time.time()

    final_centroids_2d, final_labels_2d, final_inertia_2d, iterations_2d = kmeans(
        X_data_2d, k_clusters_2d, max_iters=300, tol=1e-5
    )
    end_time = time.time()

    print(f"\n--- K-Means Results (2D) ---")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Iterations performed: {iterations_2d}")
    print(f"Final Inertia (WCSS): {final_inertia_2d:.4f}")

    plot_kmeans_results(X_data_2d, final_labels_2d, final_centroids_2d)

    print("\n" + "="*30)
    print(" K-Means Example with 5D Data ")
    print("="*30)
    k_clusters_5d = 5

    X_data_5d, y_true_5d = make_blobs(n_samples=400, centers=k_clusters_5d,
                                      n_features=5, cluster_std=1.5,
                                      random_state=123)

    print(f"\nGenerated 5D data with shape: {X_data_5d.shape}")
    print(f"Running K-Means for k={k_clusters_5d}...")

    start_time = time.time()
    final_centroids_5d, final_labels_5d, final_inertia_5d, iterations_5d = kmeans(
        X_data_5d, k_clusters_5d, max_iters=300, tol=1e-5
    )
    end_time = time.time()

    print(f"\n--- K-Means Results (5D) ---")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Iterations performed: {iterations_5d}")
    print(f"Final Inertia (WCSS): {final_inertia_5d:.4f}")

    plot_kmeans_results(X_data_5d, final_labels_5d, final_centroids_5d)

    print("\nScript finished.")

