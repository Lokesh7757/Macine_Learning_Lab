import numpy as np

points = np.array([[0.1, 0.6], [0.2, 0.3], [0.15, 0.71], [0.08, 0.9], [0.25, 0.5], [0.24, 0.1], [0.16,
0.85], [0.3, 0.2]])

centroid1 = np.array([0.1, 0.6])
centroid2 = np.array([0.3, 0.2])

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

num_iterations = 100
for _ in range(num_iterations):
    cluster1 = []
    cluster2 = []

    for point in points:
        distance_to_c1 = euclidean_distance(point, centroid1)
        distance_to_c2 = euclidean_distance(point, centroid2)
        if distance_to_c1 < distance_to_c2:
            cluster1.append(point)
        else:
            cluster2.append(point)
        
    new_centroid1 = np.mean(cluster1, axis=0)
    new_centroid2 = np.mean(cluster2, axis=0)

    if np.array_equal(new_centroid1, centroid1) and np.array_equal(new_centroid2, centroid2):
        break

centroid1 = new_centroid1
centroid2 = new_centroid2

print("Final clusters:")
print("Cluster 1:", cluster1)
print("Cluster 2:", cluster2)

point_P6 = np.array([0.24, 0.1])
distance_to_c1_P6 = euclidean_distance(point_P6, centroid1)
distance_to_c2_P6 = euclidean_distance(point_P6, centroid2)
cluster_P6 = 1 if distance_to_c1_P6 < distance_to_c2_P6 else 2
print("\na) Cluster of P6:", cluster_P6)

population_cluster2 = len(cluster2)
print("\nb) Population of cluster around m2:", population_cluster2)

print("\nc) Updated value of m1:", centroid1)
print(" Updated value of m2:", centroid2)
