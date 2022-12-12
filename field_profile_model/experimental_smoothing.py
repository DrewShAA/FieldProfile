import numpy as np
from scipy import signal
import tqdm
import cv2 as cv



def flf_matrix(series):
    matrix = []
    for row in tqdm.tqdm(series):
        matrix.append(flf(row))
    return np.array(matrix)


def flf(vector: np.ndarray):
    b, a = signal.butter(8, 0.25, 'highpass')  # Конфигурационный фильтр 8 указывает порядок фильтра
    return np.array(signal.filtfilt(b, a, vector))


def exponential_smoothing_matrix(series, alpha):
    matrix = []
    for row in series:
        matrix.append(exponential_smoothing_v(row, alpha))
    return np.array(matrix)


def exponential_smoothing_v(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


def negative_smoothing_matrix(series):
    matrix = []
    for row in series:
        matrix.append(negative_smoothing_v(row))
    return np.array(matrix)


def negative_smoothing_v(series):
    result = []
    for val in series:
        if val < 0:
            result.append(0)
        else:
            result.append(val)
    return np.array(result)


def low_f_smoothing_matrix(series):
    matrix = []
    for row in series:
        matrix.append(low_f_smoothing_v(row))
    return np.array(matrix)


def low_f_smoothing_v(series):
    result = []
    for val in series:
        if val < 25 + 1j:
            result.append(val + (100 + 1j))
        else:
            result.append(val)
    return np.array(result)


def autolevel():
    pass


def get_border_by_prewiit(img):
    kernel_x = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=int)
    kernel_y = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=int)
    x = cv.filter2D(img, cv.CV_16S, kernel_x)
    y = cv.filter2D(img, cv.CV_16S, kernel_y)
    # Turn uint8
    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)
    img_prew = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return img_prew


