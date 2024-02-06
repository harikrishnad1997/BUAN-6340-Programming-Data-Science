import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def hierarchical_clustering(data, k):
    n = len(data)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distances[i,j] = euclidean_distance(data[i], data[j])
            distances[j,i] = distances[i,j]

    clusters = [{i} for i in range(n)]
    cluster_distances = distances.copy()

    while len(clusters) > k:
        min_distance = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                distance = np.max([cluster_distances[x,y] for x in clusters[i] for y in clusters[j]])
                if distance < min_distance:
                    min_distance = distance
                    merge_i, merge_j = i, j

        clusters[merge_i] = clusters[merge_i].union(clusters[merge_j])
        del clusters[merge_j]

        for i in range(len(clusters)):
            if i != merge_i:
                distance = np.max([distances[x,y] for x in clusters[i] for y in clusters[merge_i]])
                cluster_distances[i, merge_i] = distance
                cluster_distances[merge_i, i] = distance

    return clusters

points = np.array([[8,4], [1,4], [2,2], [2,4], [3,3], [6,2], [6,4], [7,3], [8,2], [1,2]])
clusters = hierarchical_clustering(points, 2)
print(clusters)