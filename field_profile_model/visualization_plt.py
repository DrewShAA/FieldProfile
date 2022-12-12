from matplotlib import pyplot as plt
import numpy as np
import math
import cv2 as cv
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())

def visualization_plot_matrix(matrix: np.ndarray):
    shape_m = matrix.shape
    matrix_T = matrix.T
    print(shape_m)
    kernel = int(math.sqrt(shape_m[1]) + 1)
    fig, axes = plt.subplots(kernel, kernel)
    for k in range(0, shape_m[1]):
        i = k // kernel
        j = (kernel + k) % kernel
        print(i, j)
        axes[i, j].plot(matrix_T[k])
    #print(matrix)

    # plt.plot(matrix_T, color='orange')
    plt.show()


def visualization_vector(vector: np.ndarray):

    len_vector = len(vector)
    plt.scatter(range(len_vector), vector)
    plt.plot(vector, color='orange')
    plt.show()


def show_img(windows_name, image):
    cv.namedWindow(windows_name, cv.WINDOW_NORMAL)
    cv.imshow(windows_name, image)
    # key = cv.waitKey()
    # while key != ord('q'):
    #     key = cv.waitKey(1000)
