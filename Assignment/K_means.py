import numpy as np

class K_means:

    def __init__(self, initial_centroids: np.ndarray, samples: np.ndarray):
        self.k = initial_centroids.shape[0]
        self.samples = samples
        self.num_of_samples = self.samples.shape[0]
        self.centroids = initial_centroids
        self.clusters = []
        self.should_terminate = False

    def update_clusters(self):

        distances = np.zeros((self.num_of_samples, self.k))
        for i in range(self.k):
            # find the distance of each vector to each centroid
            distances[:, i] = np.linalg.norm(self.samples - self.centroids[i], axis=1)
        # find the index of the closest centroid cluster per sample of shape (num_of_samples, 1)
        self.clusters = np.argmin(distances, axis=1)

    def update_centroids(self):
        new_centroids = np.zeros(self.centroids.shape)

        for i in range(self.k):
            cluster_i = self.samples[self.clusters == i]
            if len(cluster_i) == 0 :
                new_centroids[i] = self.centroids[i]
            else:
                new_centroids[i] = np.mean(cluster_i, axis=0)

        error = np.linalg.norm(new_centroids - self.centroids)
        self.should_terminate = error == 0
        self.centroids = new_centroids

    def run(self):
        count = 0
        # start iterating
        while not self.should_terminate:
            count += 1
            self.update_clusters()
            self.update_centroids()

        return self.clusters, self.centroids