import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	# Your code here
    def euc_dist(A, B):
        return np.linalg.norm(np.asarray(A)-np.asarray(B))

    for _ in range(0,max_iterations):
        dist_mat = []
        for i in initial_centroids:
            tmp = [euc_dist(i,j) for j in points] 
            dist_mat.append(tmp)

        assignment = np.argmin(np.asarray(dist_mat), axis=0)
        new_centroid = []
        for i in range(0,k):
            sel = assignment==i
            if sum(sel) > 0:
                new = np.mean(np.asarray(points)[sel],axis=0)
                new_centroid.append(new)
            else:
                new_centroid.append(initial_centroids[k])

        initial_centroids = new_centroid
    final_centroids = np.round(initial_centroids, 4)
	return final_centroids