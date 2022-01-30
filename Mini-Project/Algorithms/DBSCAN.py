import numpy as np
import queue

class DBSCAN:
    def __init__(self, samples, epsilon, min_neighbours):
        self.samples = samples
        self.n = len(samples)
        self.epsilon = epsilon
        self.min_neighbours = min_neighbours
        self.samples_labels = np.full(self.n, -1)
        self.neighbours_lists = []

    # Find all neighbour points at epsilon distance
    def find_neighbour_points(self, i):
        neighbour_points = []
        center = self.samples[i]
        for j in range(self.n):
            if np.linalg.norm(center - self.samples[j]) <= self.epsilon and j != i:
                neighbour_points.append(j)
        return neighbour_points

    def set_neighbours_lists(self):
        for i in range(self.n):
            self.neighbours_lists.append(self.find_neighbour_points(i))


    def update_clusters(self, i):
        q = queue.Queue()
        visited = self.samples_labels[i] != -1
        is_core = len(self.neighbours_lists[i]) >= self.min_neighbours
        if is_core and not visited:
            self.samples_labels[i] = i
            q.put(i)

        while not q.empty():
            current = q.get()

            is_core = len(self.neighbours_lists[current]) >= self.min_neighbours
            if is_core:
                for neighbour in self.neighbours_lists[current]:
                    visited = self.samples_labels[neighbour] != -1
                    if not visited:
                        self.samples_labels[neighbour] = i
                        q.put(neighbour)


    def run(self):
        self.set_neighbours_lists()
        for i in range(self.n):
            self.update_clusters(i)

        cluster_number = 0
        unique_clusters = np.unique(self.samples_labels)
        for unique_cluster in unique_clusters:
            if unique_cluster != -1:
                self.samples_labels[self.samples_labels == unique_cluster] = cluster_number
                cluster_number += 1

        return self.samples_labels