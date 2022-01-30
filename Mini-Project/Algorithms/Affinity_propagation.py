import numpy as np

class Affinity_propagation:
    # if preference == None, preference as median will be used
    def __init__(self, samples, preference, max_iter, convergence_iter, damping):
        self.samples = samples
        self.samples_labels = []
        self.n = samples.shape[0]
        self.preference = preference
        self.damping = damping

        self.S = self.compute_S()
        self.A = np.zeros((self.n, self.n))
        self.R = np.zeros((self.n, self.n))

        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.iterations_without_change = 0

    def similarity(self, xi, xj):
        return -(np.linalg.norm(xi -xj))**2

    def compute_S(self):
        S = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                similarity = self.similarity(self.samples[i], self.samples[j])
                S[i,j] = similarity
                S[j,i] = similarity

        if self.preference is None:
            self.preference = np.median(S)

        np.fill_diagonal(S, self.preference)

        return S


    def update_responsibilities(self):
        S_A = self.S + self.A

        # fill diagonal with infinity values, that way a sample would not choose itself as the exampler
        np.fill_diagonal(S_A, -np.inf)

        rows_max_idx = np.argmax(S_A, axis=1)
        rows_max = S_A[range(self.n), rows_max_idx]

        # index of max value may be pointing to itself, hence, compute the next max value
        S_A[range(self.n), rows_max_idx] = -np.inf
        next_rows_max = np.max(S_A, axis=1)

        # fill max_matrix with maximum values for each row
        max_matrix = np.transpose(np.tile(rows_max, (self.n, 1)).reshape((self.n, self.n)))
        # fix maximum value of the index of the real maximum, as k=k' is not desired.
        max_matrix[range(self.n), rows_max_idx] = next_rows_max

        self.R = self.R * self.damping + (1 - self.damping) * (self.S - max_matrix)

    def update_availabilities(self):
        R_clipped = self.R.copy()
        R_clipped = np.clip(R_clipped, 0, np.inf) # only sum of positive values is needed
        np.fill_diagonal(R_clipped, 0) # diagonal is not included in sum
        A = R_clipped.copy()

        # column wise sum including elements that i=i'
        A = np.tile(A.sum(axis=0), (self.n, 1)).reshape((self.n, self.n))
        # remove R(i,k) from A(i,k) that was wrongfully added
        A -= R_clipped

        A = A + self.R[range(self.n), range(self.n)]

        # choose minimum of 0 and R(k,k) + sum(max(0, R(i',k))
        #                           (i' st. i' not in {i,k})
        A = np.clip(A, -np.inf, 0)

        # fill A diagonal with sum(max(0, R(i',k))
        #                 (i' st. i' != k)
        A[range(self.n, self.n)] = R_clipped.sum(axis=0)

        self.A = self.A * self.damping + (1 - self.damping) * A


    def run(self):
        # start iterating
        for i in range(self.max_iter):
            self.update_responsibilities()
            self.update_availabilities()

            # check for convergence iteration restriction
            old_exemplars_amount = len(np.unique(self.samples_labels))
            sol = self.A + self.R
            self.samples_labels = np.argmax(sol, axis=1)
            new_exemplars_amount = len(np.unique(self.samples_labels))
            if old_exemplars_amount == new_exemplars_amount:
                self.iterations_without_change += 1
            else:
                self.iterations_without_change = 0

            if self.iterations_without_change >= self.convergence_iter:
                break


        cluster_number = 0
        unique_clusters = np.unique(self.samples_labels)
        for unique_cluster in unique_clusters:
            self.samples_labels[self.samples_labels == unique_cluster] = cluster_number
            cluster_number += 1

        return self.samples_labels, unique_clusters