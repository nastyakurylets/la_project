import numpy as np
import random
import math

random.seed(22)
np.random.seed(22)

def euclidean_distance_sq(point1:np.ndarray, point2:np.ndarray) -> float:
    """ Calculates the squared Euclidean distance between two points. """
    res = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        res += diff * diff
    return res

def kmeans_plusplus_init(A:np.ndarray, k:int) -> np.ndarray:
    """ Initializes centroids using the K-Means++ algorithm."""
    num_points, num_dimensions = A.shape
    centroids = np.empty((k, num_dimensions)) 

    first_idx = random.randint(0, num_points - 1)
    centroids[0] = A[first_idx]
    
    min_dist_sq = [math.inf] * num_points

    for i in range(1, k):
        sum_dist_sq_total = 0.0

        last_centroid = centroids[i-1]
        for j in range(num_points):
            min_dist_sq[j] = min(min_dist_sq[j], euclidean_distance_sq(A[j], last_centroid))
            sum_dist_sq_total += min_dist_sq[j]
            
        probabilities = [(d / sum_dist_sq_total) for d in min_dist_sq]
        
        rand_val = random.random()
        cumulative_prob = 0.0
        next_centroid_idx = num_points - 1
        for j in range(num_points):
            cumulative_prob += probabilities[j]
            if rand_val <= cumulative_prob:
                next_centroid_idx = j
                break

        centroids[i] = A[next_centroid_idx]

    return centroids


def assign_clusters(A:np.ndarray, centroids:np.ndarray) -> np.ndarray:
    """ Assigns each point in A to the nearest centroid. """
    num_points, k = A.shape[0], centroids.shape[0]
    labels = np.empty(num_points, dtype=int)

    for i in range(num_points):
        min_dist = math.inf
        assigned_label = -1

        for j in range(k):
            dist = euclidean_distance_sq(A[i], centroids[j])

            if dist < min_dist:
                min_dist = dist
                assigned_label = j
        
        labels[i] = assigned_label

    return labels

def update_centroids(A:np.ndarray, labels:np.ndarray, k:int) -> np.ndarray:
    """ Updates centroids based on the current labels. """
    num_points, num_dimensions = A.shape
    new_centroids = np.empty((k, num_dimensions))

    for i in range(k):
        points_in_cluster_indices = []
        for idx in range(num_points):
            if labels[idx] == i:
                points_in_cluster_indices.append(idx)

        sum_vector = np.zeros(num_dimensions)
        for idx in points_in_cluster_indices:
            sum_vector += A[idx]

        new_centroids[i] = sum_vector / len(points_in_cluster_indices)

    return new_centroids

def wcss(A:np.ndarray, labels:np.ndarray, centroids:np.ndarray) -> float:
    """ Calculates the Within-Cluster Sum of Squares (WCSS) for the given labels and centroids. """
    num_points = A.shape[0]
    inertia = 0.0

    for i in range(num_points):
        inertia += euclidean_distance_sq(A[i], centroids[labels[i]])

    return inertia

def kmeans(A:np.ndarray, k:int, max_iters:int=100, tol:float=1e-4) -> tuple:
    """ Performs K-means using K-Means++ initialization and Lloyd's loop """
    centroids = kmeans_plusplus_init(A, k)
    
    n_iter = 0
    for i in range(max_iters):
        n_iter = i + 1
        old_centroids = centroids.copy()

        labels = assign_clusters(A, centroids)

        centroids = update_centroids(A, labels, k)

        centroid_shift_sq_total = 0.0
        for cluster_idx in range(k):
            shift_sq = euclidean_distance_sq(centroids[cluster_idx], old_centroids[cluster_idx])
            centroid_shift_sq_total += shift_sq

        if centroid_shift_sq_total < tol:
            break

    final_labels = assign_clusters(A, centroids)
    final_inertia = wcss(A, final_labels, centroids)


    return centroids, final_labels, final_inertia, n_iter