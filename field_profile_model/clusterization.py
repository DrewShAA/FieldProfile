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


def dbscan_cluster(X, shape_m, eps):
    #  0.8301886792452831
    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros)
    # nbrs = NearestNeighbors(n_neighbors=8).fit(X)
    # # Find the k-neighbors of a point
    # neigh_dist, neigh_ind = nbrs.kneighbors(X)
    # # sort the neighbor distances (lengths to points) in ascending order
    # # axis = 0 represents sort along first axis i.e. sort along row
    # sort_neigh_dist = np.sort(neigh_dist, axis=0)
    #
    # k_dist = sort_neigh_dist[:, 4]
    # plt.plot(k_dist)
    # plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
    # plt.ylabel("k-NN distance")
    # plt.xlabel("Sorted observations (4th NN)")
    # plt.show()

    # #############################################################################

    X = StandardScaler().fit_transform(X)
    print('type X', type(X))

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=3).fit(X.copy())
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

    cv.namedWindow(f'db_scan_cluster_ {eps}', cv.WINDOW_NORMAL)
    cv.imshow(f'db_scan_cluster_ {eps}', out_matrix.astype(np.uint8))

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


def clustering(x, shape_m, num_of_clust=3):
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


def get_new_shape(n: int, m: int):
    size = ((n - 2) * (m - 2)) * 8 +\
           (2 * (n - 2) + 2 * (m - 2)) * 5 +\
           12
    print('size:', size)
    print(f'отношение размеров = {n/m}')
    sqrt_ = math.sqrt(size)
    new_n = new_m = round(sqrt_)
    if new_n * new_m < size:
        new_n += 1

    print('new shape:', new_n, new_m)
    # print(f'отношение размеров = {new_n/new_m}')

    return new_n, new_m


def get_neighbors_index(arr, i, j, kernal):
    dist = kernal // 2
    bottom_brdr = arr.shape[0] - 1
    right_brdr = arr.shape[1] - 1
    for di in range(-dist, dist + 1):
        if i + di < 0:
            continue
        if i + di > bottom_brdr:
            break
        for dj in range(-dist, dist + 1):
            if di == dj == 0:
                continue
            if j + dj < 0:
                continue
            if j + dj > right_brdr:
                break
            yield i + di, j + dj

class ElevError:
    def __init__(self, elev_error, numbers):
        self.elev_error = elev_error
        self.numbers = numbers
        self.log_base_10 = math.log(numbers, 10)
        self.number_of_merge = 0

class Sigma:
    def __init__(self):
        self.value = 0
        self.members = []

    def set_item(self, elev_err: ElevError):
        self.members.append(elev_err)
        self.value += elev_err.log_base_10

    def __add__(self, other):
        self.value += other.value
        self.members += other.members
        return self


class Coefficient:
    def __init__(self, elevations: np.ndarray, points: np.ndarray):
        self.points_x_y_z = points
        self.elevations = elevations
        self.old_n, self.old_m = self.elevations.shape[:2]
        self.n, self.m = get_new_shape(self.old_n, self.old_m)
        self.k_matrix = np.full((self.n, self.m), fill_value=9999)
        self.cnt = 0
        self.glob_iter = 0
        self.kernel = 3
        self.cust_clust = None
        self.set_all_koef = []
        self.calculate_elev_error()
        self.median_err = statistics.median(k for row in self.k_matrix for k in row if k != 9999) #/ self.cnt
        self.avg_err = sum(k for row in self.k_matrix for k in row if k != 9999) / self.cnt
        print('sqrt(self.avg_err)=', math.sqrt(self.avg_err))
        self.error_matrix = abs(self.avg_err - self.k_matrix)

    def calculate_elev_error(self):
        """
            (N - X) ^ 2 + C = err
            X - каждый элемент из массива высот
            N - каждый сосед для Х на дистанции 1
            C - коэффициент, для диагональных соседей С=2,
                для прямых 1

        """
        for i in range(self.elevations.shape[0]):
            for j in range(self.elevations.shape[1]):
                for n_i, n_j in get_neighbors_index(self.elevations, i, j, 3):
                    c = 2 if n_i != i and n_j != j else 1
                    self.calcul_k(self.elevations[n_i][n_j], self.elevations[i][j], c)

    def calculate_eucl_dist(self):
        """
        """
        dists = []
        for i in range(self.old_n):
            for j in range(self.old_m):
                # temp_dist = []
                for n_i, n_j in get_neighbors_index(self.points_x_y_z, i, j, 11):
                    dists.append(clcl(self.points_x_y_z[n_i][n_j], self.points_x_y_z[i][j]))
                # dists.append(statistics.mean(temp_dist))

        dists.sort()
        return dists

    def calcul_k(self, neighbor, current, coef):
        k_i, k_j = self.get_i_j()
        self.k_matrix[k_i][k_j] = math.pow(neighbor - current, 2) + coef

    def get_i_j(self):
        k_i, k_j = self.cnt // self.n, self.cnt % self.m
        self.cnt += 1
        return k_i, k_j

    def calcul_first_hypothesis(self, depth=0):
        self.glob_iter += 1
        print('---' * 10, self.glob_iter, '---' * 10)

        min_el = self.error_matrix.min()
        min_i = min_j = None
        if min_el == 0:
            self.set_all_koef = list(set(self.error_matrix.flatten()))
            self.set_all_koef.sort()
            min_el = self.set_all_koef[1]
            for i in range(self.error_matrix.shape[0]):
                for j in range(self.error_matrix.shape[1]):
                    if self.error_matrix[i][j] == 0:
                        self.error_matrix[i][j] = self.set_all_koef[1]

        colect = collections.Counter(list(self.error_matrix.flatten()))
        if self.cust_clust is None:
            self.cust_clust = dict(colect)
        else:
            if self.cust_clust == dict(colect):
                self.kernel += 2
                print('new_dist =', self.kernel // 2)
            else:
                self.cust_clust = dict(colect)
        print(colect)
        cnt_find = random.randrange(0, colect[min_el])

        num_of_find = 0
        for i, row in enumerate(self.error_matrix):
            is_find = False
            if min_el not in row:
                continue
            for j, el in enumerate(row):
                if el == min_el:
                    min_i = i
                    min_j = j
                    if num_of_find == cnt_find:
                        is_find = True
                        break
                    else:
                        num_of_find += 1
            if is_find:
                break

        print(f'self.error_matrix[{min_i}][{min_j}] = {self.error_matrix[min_i][min_j]};'
              f' real min = {self.error_matrix.min()}')
        # if not (0 < min_i < self.error_matrix.shape[0] - 1 and 0 < min_j < self.error_matrix.shape[1] - 1):
        #     print('граничный случай: ', min_i, min_j)
        #     return
        neighbors = []
        for n_i, n_j in get_neighbors_index(self.error_matrix, min_i, min_j, self.kernel):
            neighbors.append(self.error_matrix[n_i][n_j])

        if not neighbors:
            print('соседей нет')
            return

        if min(neighbors) == max(neighbors):
            return self.calcul_first_hypothesis(depth=depth)

        neighbors.sort(reverse=True)
        index = int(len(neighbors) * 0.5)
        # index += 1 if len(neighbors) % 2 != 0 else 0
        while neighbors[index] == neighbors[-1]:
            index -= 1

        e1 = neighbors[-1] / min_el
        e2 = neighbors[index] / min_el

        if e1 == e2 == 1:
            return

        if depth % 3 == 0:
            # проход по строчно
            x_0 = self.error_matrix[0][0]
            for i in range(self.error_matrix.shape[0]):
                for j in range(self.error_matrix.shape[1]):
                    if e1 < self.error_matrix[i][j] / x_0 < e2:
                        self.error_matrix[i][j] = x_0
                        continue
                    x_0 = self.error_matrix[i][j]
        elif depth % 3 == 1:
            # проход по столбцам
            x_0 = self.error_matrix[0][0]
            for j in range(self.error_matrix.shape[1]):
                for i in range(self.error_matrix.shape[0]):
                    if e1 < self.error_matrix[i][j] / x_0 < e2:
                        self.error_matrix[i][j] = x_0
                        continue
                    x_0 = self.error_matrix[i][j]
        else:
            # проход по диагонали
            x_0 = self.error_matrix[-1][0]
            for i in range(self.error_matrix.shape[0], 0, -1):
                for j in range(0, self.error_matrix.shape[1] - i):
                    if e1 < self.error_matrix[i][j] / x_0 < e2:
                        self.error_matrix[i][j] = x_0
                        continue
                    x_0 = self.error_matrix[i][j]
        return self.calcul_first_hypothesis(depth=depth+1)

    def calcul_second_hypothesis(self):
        error_collection = dict(collections.Counter(list(self.error_matrix.flatten())))

        elevation_error = union_error_by_absolut_dist(error_collection)

        sigmas = self.group_elev_error(elevation_error)

        max_sigma = max(sigmas, key=lambda s: s.value)
        sigmas.remove(max_sigma)
        while len(sigmas) > 1:
            sum_sigm = sum(map(lambda sigma: sigma.value, sigmas))
            if max_sigma.value / sum(map(lambda sigma: sigma.value, sigmas)) >= 0.4:
                break
            if len(sigmas) < 2:
                print('не нашли результат сигмы кончились')
                break
            temp_max = max(sigmas, key=lambda s: s.value)
            sigmas.remove(temp_max)
            max_sigma += temp_max

        winners = max_sigma.members
        losers = [member for sigma in sigmas for member in sigma.members]
        print('l numbers before', list(map(lambda x: {x.elev_error: x.numbers}, losers)))
        print('w numbers before', list(map(lambda x: {x.elev_error: x.numbers}, winners)))
        print('---')
        while losers:
            loser = losers[0]
            for winner in winners:
                winner.ratio_win_los = 0
                winner.ratio_win_los = abs(winner.elev_error / loser.elev_error - 1)

            min_rat_winner = winners[0]
            min_rat = min_rat_winner.ratio_win_los
            for winner in winners[1:]:
                if min_rat < winner.ratio_win_los:
                    continue
                min_rat_winner = winner
                min_rat = min_rat_winner.ratio_win_los

            del losers[0]
            min_rat_winner.numbers += loser.numbers
            min_rat_winner.number_of_merge += 1
        print('w numbers after ', list(map(lambda x: {x.elev_error: x.numbers}, winners)))

        return len(winners)

    def group_elev_error(self, el_ers: List[ElevError]):
        border = 0.99
        step = 1
        sigmas = [Sigma()]
        for el_er in el_ers:
            if el_er.log_base_10 > border:
                border += step
                sigmas.append(Sigma())
            sigmas[-1].set_item(el_er)
        return sigmas


def clcl(p_1, p_2):
    dist = math.sqrt(sum(pow(p_2[i] - p_1[i], 2) for i in range(len(p_2))))
    return dist


def union_error_by_absolut_dist(error_collection: dict):
    keys = list(error_collection.keys())
    for key in keys:
        if key > 2000:
            error_collection.pop(key)

    new_error_collection = {}
    while True:
        is_break_w = False
        keys = list(error_collection.keys())
        for f_k_i in range(len(keys)):
            is_break = False
            for s_k_i in range(f_k_i + 1, len(keys)):
                if abs(keys[f_k_i] - keys[s_k_i]) >= 0.1:
                    continue
                f_v = error_collection.pop(keys[f_k_i])
                s_v = error_collection.pop(keys[s_k_i])
                new_error_collection[(keys[f_k_i] + keys[s_k_i]) / 2] = f_v + s_v
                is_break = True
                break
            if is_break:
                break
        else:
            is_break_w = True
        if is_break_w:
            break

    for key, value in error_collection.items():
        new_error_collection[key] = value

    elevation_error = [ElevError(*data) for data in new_error_collection.items()]
    elevation_error.sort(key=operator.attrgetter('log_base_10'))
    return elevation_error


def get_hyperparams_for_clusterization(elevations: np.ndarray):
    old_n, old_m = elevations.shape[:2]
    n, m = get_new_shape(old_n, old_m)
    k_matrix = np.full((n, m), fill_value=9999)


def main():
    for i in range(10):
        if i == 5:
            break
    else:
        print('all good')
    res = math.log(1001, 10)
    print(res)


def dist_pairwise(matrix_in_m):
    a = matrix_in_m.copy().reshape(-1, 2)
    b = a.copy()
    P = np.add.outer(np.sum(a**2, axis=1), np.sum(b**2, axis=1))
    N = np.dot(a, b.T)
    res = np.sqrt(P - 2 * N)
    avg_dist = sum([el for row in res for el in row]) / (res.shape[0] * res.shape[1])

    return avg_dist

if __name__ == '__main__':
    main()
