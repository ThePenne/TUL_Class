from Algorithms import Affinity_propagation as our_AP, Hierarchical_clustering as our_HC, DBSCAN as our_DBSCAN
from sklearn.cluster import AffinityPropagation as sk_AP, AgglomerativeClustering as sk_HC, DBSCAN as sk_DBSCAN

from sklearn.datasets import  make_circles, make_moons, make_blobs, make_classification
from matplotlib import pyplot as plt
from itertools import cycle
import numpy as np

def plot_Data(samples, samples_labels, title):
    clusters = np.unique(samples_labels)
    COLOR_CYCLE = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                         "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
    plt.title(f"The {title} generated data:")
    colors = dict(zip(clusters, COLOR_CYCLE))
    for i in range(len(clusters)):
        plt.scatter(samples[samples_labels == clusters[i], 0], samples[samples_labels == clusters[i], 1], s=20,
                    c=colors.get(clusters[i]))
    plt.show()

def plot_DBSCAN(algorithm, samples, samples_labels, epsilon, min_neighbors):
    plt.title(f"{algorithm}: DBSCAN with {min_neighbors} min-neighbors and \u03B5={epsilon}")

    clusters = np.unique(samples_labels)
    COLOR_CYCLE = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                         "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
    colors = dict(zip(np.delete(clusters, np.where(clusters == -1)), COLOR_CYCLE))
    colors.update({-1: 'k'})

    for i in range(len(clusters)):
        plt.scatter(samples[samples_labels == clusters[i], 0], samples[samples_labels == clusters[i], 1],
                    s=20, c=colors.get(clusters[i]))

    plt.show()

def plot_HC(algorithm, samples, samples_labels, linkage, k):
    plt.title(f"{algorithm}: hierarchical clustering with {linkage} linkage and k={k}")
    clusters = np.unique(samples_labels)

    COLOR_CYCLE = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                         "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
    colors = dict(zip(clusters, COLOR_CYCLE))

    for i in range(len(clusters)):
        plt.scatter(samples[samples_labels == clusters[i], 0], samples[samples_labels == clusters[i], 1], s=20,
                    c=colors.get(clusters[i]))

    plt.show()

def plot_AP(algorithm, samples, samples_labels, exemplars, iter, preference, damping):
    plt.title(f"{algorithm}: affinity propagation after {iter} iterations\nwith preference={preference} and damping={damping}.\nreceived {len(exemplars)} exemplars")

    COLOR_CYCLE = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                         "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
    colors = dict(zip(exemplars, COLOR_CYCLE))

    sample_size = len(samples_labels)
    # fix labels for printing
    for i in range(sample_size):
        samples_labels[i] = exemplars[samples_labels[i]]

    for i in range(sample_size):
        if i not in exemplars:
            plt.scatter(samples[i][0], samples[i][1], s=20, c=colors[samples_labels[i]])

    for exemplar in exemplars:
        plt.scatter(samples[exemplar][0], samples[exemplar][1], s=50, edgecolors='k', c=colors[exemplar])


    plt.show()


def generate_dataset(sample_size):

    noisy_circles, noisy_circles_labels = make_circles(n_samples=sample_size, factor=0.5, noise=0.05)
    plot_Data(noisy_circles, noisy_circles_labels, "noisy circles")

    noisy_moons, noisy_moons_labels = make_moons(n_samples=sample_size, noise=0.05, random_state=2)
    plot_Data(noisy_moons, noisy_moons_labels, "noisy moons")

    blobs, blobs_labels = make_blobs(n_samples=sample_size, random_state=38)
    plot_Data(blobs, blobs_labels, "blobs")

    gaussians, gaussians_labels = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                                                      n_clusters_per_class=1, random_state=4)
    plot_Data(gaussians, gaussians_labels, "gaussians")

    return noisy_circles, noisy_moons, blobs, gaussians


def compare_DBSCAN(samples, epsilon, min_neighbors):
    dbscan1 = our_DBSCAN.DBSCAN(samples, epsilon, min_neighbors)
    samples_labels1 = dbscan1.run()
    plot_DBSCAN("OUR DBSCAN", samples, samples_labels1, epsilon, min_neighbors)

    dbscan2 = sk_DBSCAN(eps=epsilon, min_samples=min_neighbors)
    samples_labels2 = dbscan2.fit_predict(samples)
    plot_DBSCAN("SK DBSCAN", samples, samples_labels2, epsilon, min_neighbors)

def compare_HC(samples, linkage, k):
    hc1 = our_HC.hierarchical_clustering(samples, linkage, k)
    samples_labels1 = hc1.run()
    plot_HC("OUR HC", samples, samples_labels1, linkage, k)

    hc2 = sk_HC(linkage=linkage, n_clusters=k)
    samples_labels2 = hc2.fit_predict(samples)
    plot_HC("SK HC", samples, samples_labels2, linkage, k)

def compare_AP(samples, preference, iter, convergence_iter, damping):
    ap1 = our_AP.Affinity_propagation(samples, preference, iter, convergence_iter, damping)
    samples_labels1, exemplars1 = ap1.run()
    plot_AP("OUR AP", samples, samples_labels1, exemplars1, iter, preference, damping)

    ap2 = sk_AP(preference=preference, max_iter=iter, convergence_iter=convergence_iter, damping=damping, random_state=0)
    samples_labels2 = ap2.fit_predict(samples)
    exemplars2 = ap2.cluster_centers_indices_
    plot_AP("SK AP", samples, samples_labels2, exemplars2, iter, preference, damping)

if __name__ == '__main__':
    noisy_circles, noisy_moons, blobs, gaussians = generate_dataset(1500)

    # -- DBSCAN --
    compare_DBSCAN(noisy_circles, 0.1, 10)
    compare_DBSCAN(noisy_moons, 0.2, 20)
    compare_DBSCAN(blobs, 0.4, 8)
    compare_DBSCAN(gaussians, 0.3, 10)

    # -- Hierarchical Clustering --
    compare_HC(noisy_circles, 'single', 2)
    compare_HC(noisy_circles, 'complete', 2)
    compare_HC(noisy_moons, 'single', 2)
    compare_HC(noisy_moons, 'complete', 2)
    compare_HC(blobs, 'single', 3)
    compare_HC(blobs, 'complete', 3)
    compare_HC(gaussians, 'single', 2)
    compare_HC(gaussians, 'complete', 2)

    # -- Affinity Propagation --
    compare_AP(noisy_circles, None, 200, 25, 0.85) # preference in pdf = -260
    compare_AP(noisy_moons, None, 200, 25, 0.9) # preference in pdf = -200
    compare_AP(blobs, None, 200, 25, 0.9) # preference in pdf = -500
    compare_AP(gaussians, None, 200, 25, 0.9) # preference in pdf = -600
