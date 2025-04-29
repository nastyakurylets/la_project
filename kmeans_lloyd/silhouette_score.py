import math
import numpy as np
import matplotlib.pyplot as plt
from kmeans_lloyd.kmeans import euclidean_distance_sq, kmeans

def calculate_a_basic(point_index: int, A: np.ndarray, labels: np.ndarray, current_label: int) -> float:
    """ Basic calculation of average distance to other points in the same cluster. """
    points_in_same_cluster = A[labels == current_label]
    points_in_same_cluster = points_in_same_cluster[~np.all(points_in_same_cluster == A[point_index], axis=1)]

    if len(points_in_same_cluster) == 0:
        return 0.0

    total_distance = 0.0
    for other_point in points_in_same_cluster:
        total_distance += math.sqrt(euclidean_distance_sq(A[point_index], other_point))

    return total_distance / len(points_in_same_cluster)

def calculate_b_basic(point_index: int, A: np.ndarray, labels: np.ndarray, current_label: int) -> float:
    """ Basic calculation of minimum average distance to points in a different cluster. """
    unique_labels = np.unique(labels)
    other_labels = unique_labels[unique_labels != current_label]

    if len(other_labels) == 0:
        return 0.0

    min_avg_distance = math.inf

    for other_label in other_labels:
        points_in_other_cluster = A[labels == other_label]
        if len(points_in_other_cluster) == 0:
            continue

        total_distance = 0.0
        for other_point in points_in_other_cluster:
            total_distance += math.sqrt(euclidean_distance_sq(A[point_index], other_point))

        avg_distance = total_distance / len(points_in_other_cluster)
        min_avg_distance = min(min_avg_distance, avg_distance)

    return min_avg_distance

def silhouette_score_basic(A: np.ndarray, labels: np.ndarray) -> float:
    """ Basic calculation of average silhouette score. """
    num_points = A.shape[0]
    unique_labels = np.unique(labels)

    if num_points <= 1 or len(unique_labels) <= 1:
        return 0.0

    silhouette_scores = []

    for i in range(num_points):
        current_label = labels[i]
        a_i = calculate_a_basic(i, A, labels, current_label)
        b_i = calculate_b_basic(i, A, labels, current_label)

        if max(a_i, b_i) == 0:
             silhouette_i = 0.0
        else:
            silhouette_i = (b_i - a_i) / max(a_i, b_i)

        silhouette_scores.append(silhouette_i)

    return np.mean(silhouette_scores)