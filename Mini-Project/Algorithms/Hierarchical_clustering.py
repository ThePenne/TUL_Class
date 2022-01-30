import numpy as np

class hierarchical_clustering:
    def __init__(self, samples, linkage, k):
        self.samples = samples
        self.n = samples.shape[0]
        if linkage == 'single':
            self.update_next_linkage = self.single_linkage
        elif linkage == 'complete':
            self.update_next_linkage = self.complete_linkage

        self.k = k
        self.dist_matrix = self.compute_dist_matrix()
        self.samples_labels = np.array(range(self.n))
        self.min_dist_idx = [-1, -1]

    def compute_dist(self, xi, xj):
        return np.linalg.norm(xi -xj)

    def compute_dist_matrix(self):
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                dist = self.compute_dist(self.samples[i], self.samples[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
            # fill diagonal with infinity values, that way clusters would not choose itself for combining
            dist_matrix[i, i] = np.inf

        return dist_matrix


    def single_linkage(self):
        # merge 'b' cluster to 'a' cluster
        a = self.min_dist_idx[0]
        b = self.min_dist_idx[1]

        for i in range(self.n):
            if (i != a and i != b):
                temp = min(self.dist_matrix[a][i], self.dist_matrix[b][i])
                self.dist_matrix[a][i] = temp
                self.dist_matrix[i][a] = temp

        # 'b' cluster merged into 'a'. Set dist from 'b' cluster to all other clusters to be infinity
        self.dist_matrix[b, :] = np.inf
        self.dist_matrix[:, b] = np.inf

    def complete_linkage(self):
        # merge 'b' cluster to 'a' cluster
        a = self.min_dist_idx[0]
        b = self.min_dist_idx[1]

        for i in range(self.n):
            if (i != a and i != b):
                temp = max(self.dist_matrix[a][i], self.dist_matrix[b][i])
                self.dist_matrix[a][i] = temp
                self.dist_matrix[i][a] = temp

        # 'b' cluster merged into 'a'. Set dist from 'b' cluster to all other clusters to be infinity
        self.dist_matrix[b, :] = np.inf
        self.dist_matrix[:, b] = np.inf

    def update_min_dist(self):
        self.min_dist_idx =  np.unravel_index(np.argmin(self.dist_matrix), (self.n, self.n))

    def update_clusters(self, i):
        self.update_min_dist()
        self.update_next_linkage()

        # Manipulating the dictionary to keep track of cluster formation in each step
        cluster_a = self.samples_labels[self.min_dist_idx[0]]
        cluster_b = self.samples_labels[self.min_dist_idx[1]]
        self.samples_labels[self.samples_labels == cluster_a] = self.n + i
        self.samples_labels[self.samples_labels == cluster_b] = self.n + i


    def run(self):
        # start from iteration number 0, and stop k iterations from the end
        for i in range(self.n - self.k):
            self.update_clusters(i)

        cluster_number = 0
        unique_clusters = np.unique(self.samples_labels)
        for unique_cluster in unique_clusters:
            self.samples_labels[self.samples_labels == unique_cluster] = cluster_number
            cluster_number += 1

        return self.samples_labels