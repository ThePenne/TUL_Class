import numpy as np
from numpy import asarray

import matplotlib.pyplot as plt
from matplotlib import image

from K_means import K_means
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans



def plot(samples, centroids, clusters, centers, label):
    plt.title(label)
    plt.scatter(samples[:, 0], samples[:, 1], color='blue', marker="o", linestyle="None", label="samples",
                alpha=0.5, zorder=1)
    for i in range(len(samples)):
        x = [samples[i][0], centroids[clusters[i]][0]]
        y = [samples[i][1], centroids[clusters[i]][1]]
        plt.plot(x, y, color='lightcoral', alpha=0.2, zorder=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', edgecolors='black', marker="^", linestyle="None",
                label="final centroids", zorder=3)
    plt.scatter(centers[:, 0], centers[:, 1], color='yellow', edgecolors='black', marker="*", linestyle="None", label="real centroids",
                zorder=2, s=100)
    plt.legend()
    plt.show()

def resotre(centroids, clusters, shape, label):
    plt.title(label)
    output_image = np.zeros((len(clusters),3), int)
    for pixel in range(len(clusters)):
        for color in range(3):
            output_image[pixel][color] = centroids[clusters[pixel]][color]

    output_image = output_image.reshape(shape)

    plt.imshow(output_image)
    plt.show()

def main():

    # # ------------------------- comparing our implementations vs sklearn implementation: --------------------
    #
    # # generate N=1000 points from 3 isotropic Gaussians:
    # samples, labels, centers = make_blobs(n_samples=1000, centers=3, n_features=2, center_box=(0.0, 100.0),
    #                                              cluster_std=5, return_centers=True)
    #
    # k = 3
    #
    # # generate initial centroids:
    # initial_centroids = np.random.rand(k, 2) * 100
    #
    # # run k means
    # clusters, centroids = K_means(initial_centroids, samples).run()
    #
    # # plot and scatter result
    # plot(samples, centroids, clusters, centers, "k means with our implementation")
    #
    # km_sameInitials = KMeans(n_clusters=k, init=initial_centroids)
    # clusters = km_sameInitials.fit_predict(samples)
    # centroids = km_sameInitials.cluster_centers_
    #
    # # plot and scatter result
    # plot(samples, centroids, clusters, centers, "k means with sklearn implementation - same initial centroinds")
    #
    # km_smart = KMeans(n_clusters=k)
    # clusters = km_smart.fit_predict(samples)
    # centroids = km_smart.cluster_centers_
    #
    # # plot and scatter result
    # plot(samples, centroids, clusters, centers, "k means with sklearn implementation - smart initial centroinds")
    #
    # # ----------------------------- comparing different random initializations: ------------------------------
    #
    # # generate N=1000 points from 3 isotropic Gaussians:
    # samples, labels, centers = make_blobs(n_samples=1000, centers=3, n_features=2, center_box=(0.0, 100.0),
    #                                       cluster_std=5, return_centers=True)
    #
    # k = 3
    #
    # # generate initial centroids:
    # initial_centroids = np.random.rand(k, 2) * 100
    #
    # # run k means
    # clusters, centroids = K_means(initial_centroids, samples).run()
    #
    # # plot and scatter result
    # plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], color='lightgreen', s=70, edgecolors='black', marker="s", linestyle="None",
    #             label="initial centroids", zorder=3)
    # plot(samples, centroids, clusters, centers, "k means with our implementation")
    #
    # # generate another initial centroids:
    # initial_centroids = np.random.rand(k, 2) * 100
    #
    # # run k means
    # clusters, centroids = K_means(initial_centroids, samples).run()
    #
    # # plot and scatter result
    # plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], color='lightgreen', s=70, edgecolors='black', marker="s", linestyle="None",
    #             label="initial centroids", zorder=3)
    # plot(samples, centroids, clusters, centers, "k means with our implementation")
    #
    #
    # # --------------------------------- comparing different k's on the results: --------------------------------
    #
    # # generate N=1000 points from 3 isotropic Gaussians:
    # samples, labels, centers = make_blobs(n_samples=1000, centers=3, n_features=2, center_box=(0.0, 100.0), cluster_std=5, return_centers=True)
    #
    # k_vals = [1, 2, 3, 5, 20, 100, 500, 1000]
    # for k in k_vals:
    #     # generate initial centroids:
    #     initial_centroids = np.random.rand(k,2) * 100
    #     clusters, centroids = K_means(initial_centroids, samples).run()
    #
    #     plot(samples, centroids, clusters, centers, f"centroids for k = {k}")

    # ---------------------------------------------------- Mandrill ---------------------------------------------

    # load image as pixel array
    original_mandrill = image.imread('Mandrill.jpg')
    # summarize shape of the pixel array
    print(f"shape of original mandrill picture: {original_mandrill.shape}")
    # display the array of pixels as an image
    plt.title("original mandrill")
    plt.imshow(original_mandrill)
    plt.show()

    samples_red =  asarray(original_mandrill[:,:,0]).flatten()
    samples_green = asarray(original_mandrill[:,:,1]).flatten()
    samples_blue = asarray(original_mandrill[:,:,2]).flatten()

    samples = np.stack((samples_red, samples_green, samples_blue)).transpose()

    k_vals = [1, 2, 3, 4, 5, 10, 15]
    for k in k_vals:
        # generate RGB values:
        centers = np.full((k, 3), np.random.randint(0, 256))
        clusters, centroids = K_means(centers, samples).run()
        resotre(centroids.astype(int), clusters, original_mandrill.shape, f"restored mandrill image from {k}_Means")

if __name__ == '__main__':
    main()