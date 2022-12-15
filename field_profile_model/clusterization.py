import math
import collections
import operator
import random
from typing import List

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
import statistics

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())


def dbscan_cluster(X, shape_m):

    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros)
    nbrs = NearestNeighbors(n_neighbors=100).fit(X)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(X)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, 4]
    plt.plot(k_dist)
    plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (4th NN)")
    plt.show()

    # #############################################################################

    X = StandardScaler().fit_transform(X)
    print('type X', type(X))

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=42, min_samples=6).fit(X.copy())
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    clusters_matrix = labels.reshape(shape_m).astype(np.float64)  # * 255
    # clusters_matrix = np.flip(clusters_matrix)
    out_matrix = some_scaling(clusters_matrix)

    cv.namedWindow(f'db_scan_cluster', cv.WINDOW_NORMAL)
    cv.imshow(f'db_scan_cluster', out_matrix.astype(np.uint8))

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def mean_shift(X, shape_m):
    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X_, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
    print('print(X_)', X_)
    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    print('bandwidth', bandwidth)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    print(f'mean_shift labels: ', *labels)
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    clusters_matrix = labels.reshape(shape_m).astype(np.float64)  # * 255
    # clusters_matrix = np.flip(clusters_matrix)
    out_matrix = some_scaling(clusters_matrix)

    cv.namedWindow('mean_shift', cv.WINDOW_NORMAL)
    cv.imshow('mean_shift', out_matrix.astype(np.uint8))


def run_agglomerative_clusterization(x, shape_m, n_clusters, distance=None) -> np.ndarray:
    # type_aggl_clust = ['ward', 'complete', 'average']
    # for type_a in type_aggl_clust:
    #     for n_clusters in range(3, 10):
    #         ward = AgglomerativeClustering(n_clusters=n_clusters, linkage=type_a)
    #         ward.fit(x)
    #
    #         clusters_matrix = ward.labels_.reshape(shape_m).astype(np.float64)  # * 255
    #         # clusters_matrix = np.flip(clusters_matrix)
    #         out_matrix = some_scaling(clusters_matrix)
    #
    #         cv.namedWindow(type_a + str(n_clusters), cv.WINDOW_NORMAL)
    #         cv.imshow(type_a + str(n_clusters), out_matrix.astype(np.uint8))

    ward = AgglomerativeClustering(n_clusters=None if distance is not None else n_clusters,
                                   distance_threshold=distance,
                                   linkage='average')
    ward.fit(x)

    clusters_matrix = ward.labels_.reshape(shape_m).astype(np.float64)  # * 255
    p_n = str(distance)
    cv.namedWindow('AC: '+p_n, cv.WINDOW_NORMAL)
    cv.imshow('AC: '+p_n, clusters_matrix.astype(np.uint8))
    scale_mtrx = some_scaling(clusters_matrix)

    cv.namedWindow('AC_scale: '+p_n, cv.WINDOW_NORMAL)
    cv.imshow('AC_scale: '+p_n, scale_mtrx.astype(np.uint8))

    # print("out", clusters_matrix)
    print(f'clusters for distance_threshold = {p_n}', len(list(set(ward.labels_))))

    # mean_shift(x, shape_m)


def run_agglomerative_clusterization(x, shape_m, num_of_clust=3):
    ward = AgglomerativeClustering(n_clusters=num_of_clust, linkage='ward')
    ward.fit(x)

    clusters_matrix = ward.labels_.reshape(shape_m).astype(np.float64)  # * 255

    return clusters_matrix, list(set(ward.labels_))


def some_scaling(matrix: np.ndarray):
    """
    Растягивает значения в матрице от 0 до 255
    возвращает новый массив
    """
    temp_matrix = matrix.copy()

    min_evel = matrix.min()
    temp_matrix -= min_evel
    max_evel = temp_matrix.max()

    temp_matrix /= max_evel
    temp_matrix *= 255
    return temp_matrix
