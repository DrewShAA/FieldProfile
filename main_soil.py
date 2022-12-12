import argparse
import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import statistics

import field_profile_model.post_processing_cluster as ppc
from field_profile_model.read_from_csv import get_datas
from field_profile_model import clusterization


def get_list_of_data(points):
    latitudes = []
    longitude = []
    elevations = []

    for point in points:
        if point.x not in latitudes:
            latitudes.append(point.x)
        if point.y not in longitude:
            longitude.append(point.y)
        if point.z not in elevations:
            elevations.append(point.z)
    return latitudes, longitude, elevations


def create_matrix(points):
    latitudes, longitude, elevations = get_list_of_data(points)

    profile = np.zeros((len(latitudes), len(longitude)))
    x_3d = []
    y_3d = []
    z_3d = []
    list_of_list = []
    point_cloud = []

    for point in points:
        profile[latitudes.index(point.x)][longitude.index(point.y)] = point.z
        # x_3d.append(latitudes.index(point.x))
        x_3d.append(point.x_m)
        # y_3d.append(longitude.index(point.y))
        y_3d.append(point.y_m)
        z_3d.append(point.z)
        point_cloud.append((point.x_m, point.y_m, point.z))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(list(range(len(latitudes))), list(range(len(longitude))), elevations)
    ax.scatter(x_3d, y_3d, z_3d)
    plt.show()

    pdata = pyvista.PolyData(point_cloud)
    pdata.plot()

    # ## построение плоскости

    # import pandas as pd
    # df2 = pd.DataFrame(point_cloud, columns=['Latitude', 'Longitude', 'Elevation'])
    #
    # import statsmodels.formula.api as smf
    # model = smf.ols(formula='Elevation ~ Latitude + Longitude', data=df2)
    # res = model.fit()


    # ##

    p = pyvista.Plotter()
    p.add_mesh(pdata, scalars=z_3d, cmap="terrain", smooth_shading=True)
    p.show()

    matrix_of_list = np.full((len(latitudes), len(longitude)), fill_value=None)
    matrix_in_m = np.full((len(latitudes), len(longitude)), fill_value=None)
    for point in points:
        matrix_of_list[latitudes.index(point.x)][longitude.index(point.y)] = (point.x, point.y, point.z)
        matrix_in_m[latitudes.index(point.x)][longitude.index(point.y)] = [point.x_m, point.y_m, point.z]
        list_of_list.append((point.x_m, point.y_m, point.z))

    # print(list_of_list)
    #
    # print(matrix_of_list.shape)
    #
    # print('---' * 25)
    # #print(*matrix_of_list, sep='\n')
    # print('---' * 25)
    list_of_list.reverse()
    # print('new list = ', list_of_list)
    return profile, list_of_list, matrix_of_list, matrix_in_m


def get_km_by_corner_latitude(corner):
    return abs(111.321 * math.cos(corner) - 0.094 * math.cos(3 * corner))


def get_km_by_corner_longitude(corner):
    return abs(111.143 - 0.562 * math.cos(2 * corner))


def calculate_distance(points):
    min_x = min(map(lambda point: point.x, points))
    min_y = min(map(lambda point: point.y, points))
    one_angle_in_km_latitude = get_km_by_corner_latitude(min_x)
    one_angle_in_km_longitude = get_km_by_corner_longitude(min_y)
    for point in points:
        point.x_m = int(one_angle_in_km_latitude * abs(min_x - point.x) * 1000)
        point.y_m = int(one_angle_in_km_longitude * abs(min_y - point.y) * 1000)


def create_train_test_2(points):
    # [[0, 0, 209], [0, 30, 207], [0, 61, 208], [0, 92, 209], [0, 123, 212], [0, 154, 214], [0, 185, 215]
    train_x = []
    train_y = []
    test_x = []
    position = []
    for point in points:
        if point[2] == 0:
            test_x.append(point[:2])
            continue
        train_x.append(point[:2])
        train_y.append(point[2])
    return train_x, train_y, test_x, position


def main():
    # points = get_datas('samples/example_2/Results_elevation_ex_2.csv')
    points = get_datas('samples/example_1/Results_elevation_ex_1.csv')
    number_of_clusters = 3
    calculate_distance(points)

    profile, list_to_cluster, matrix_points, matrix_in_m = create_matrix(points)

    matrix_clust, clusts = clusterization.run_agglomerative_clusterization(list_to_cluster, profile.shape[:2],
                                                                           number_of_clusters)
    # clusterization.clustering(list_to_cluster, profile.shape[:2])
    # clusterization.dbscan_cluster(list_to_cluster, profile.shape[:2])

    cv.namedWindow('input', cv.WINDOW_NORMAL)
    cv.imshow('input', profile.astype(np.uint8))

    scale_img = clusterization.some_scaling(profile)
    cv.namedWindow('input_scale', cv.WINDOW_NORMAL)
    cv.imshow('input_scale', scale_img.astype(np.uint8))

    cv.namedWindow('matrix_clust', cv.WINDOW_NORMAL)
    cv.imshow('matrix_clust', matrix_clust.astype(np.uint8))

    matrix_clust_scale = clusterization.some_scaling(matrix_clust)
    cv.namedWindow('matrix_clust_scale', cv.WINDOW_NORMAL)
    cv.imshow('matrix_clust_scale', matrix_clust_scale.astype(np.uint8))

    scale_clust = ppc.select_clusters_type(matrix_clust_scale)
    # clusters_obj = ppc.area_selection(matrix_clust, clusts, profile)
    clusters_obj = ppc.area_selection(matrix_clust_scale, scale_clust, profile)
    plot_points_x = []
    plot_points_y = []
    plot_points_z = []
    result_p = []
    scalars = []
    lat = []
    long = []
    for obj in clusters_obj:
        obj.set_point_m(matrix_in_m)
        obj.area_point_with_predict_point, xs, ys, zs, angle, equation = ppc.create_train_test(obj.area_points)
        obj.angle = angle
        obj.equation = equation
        plot_points_x.append(xs)
        plot_points_y.append(ys)
        plot_points_z.append(zs)
        # obj.zs = zs
        # obj.xs = xs
        # obj.ys = ys
        result_p += [[x, y, z] for xr, yr, zr in zip(xs, ys, zs) for x, y, z in zip(xr, yr, zr)]
        scalars += [z for zr in zs for z in zr]
        lat += [z for zr in xs for z in zr]
        long += [z for zr in ys for z in zr]
        pdata = pyvista.PolyData(obj.get_list_of_points())
        p = pyvista.Plotter()
        p.add_mesh(pdata, scalars=obj.area_points[..., 2], cmap="terrain", smooth_shading=True)
        p.show(title=f'angle = {angle}')

        data_for_poly = [list(el) for row in obj.area_point_with_predict_point for el in row]
        pdata = pyvista.PolyData(data_for_poly)
        p = pyvista.Plotter()
        p.add_mesh(pdata, scalars=obj.area_point_with_predict_point[..., 2], cmap="terrain", smooth_shading=True)
        p.show(title=f'angle = {angle}')

    for obj in clusters_obj:
        print(f'equation: {obj.equation}; angle: {obj.angle}')
    print('avg angle = ', statistics.mean(map(lambda x: x.angle, clusters_obj)))
    pdata = pyvista.PolyData(result_p)
    p = pyvista.Plotter()
    p.add_mesh(pdata, scalars=scalars, cmap="terrain", smooth_shading=True)
    p.show()

    surf = pdata.delaunay_2d()
    surf.plot(show_edges=False)

    grid = pyvista.StructuredGrid(np.array(lat), np.array(long), np.array(scalars))
    grid.plot()

    p = pyvista.Plotter()
    p.add_mesh(pdata, cmap="terrain", smooth_shading=True)
    p.show()


if __name__ == "__main__":
    print('start')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, default="/samples/Results_elevation_ex_1.csv",
                        help='Путь к файлу с высотами')

    parser.parse_args()
    main()
    print('done.')
