import random
import time
import statistics
import cv2 as cv
from PIL import Image, ImageFilter, ImageOps, ImageShow
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())


def show_img(windows_name, image):
    cv.namedWindow(windows_name, cv.WINDOW_NORMAL)
    cv.imshow(windows_name, image)


def vector_smoothing(vector_np: np.ndarray):
    vector: np.ndarray = vector_np.copy()
    vector = vector.real.astype(dtype=np.uint8)
    print(dir(vector))
    print('min', vector.min())
    print('max', vector.max())
    print('matrix mian', vector.mean())
    print('median', statistics.median(vector))
    print('avg', statistics.mean(vector))
    # print('real', vector[:100])
    # print('complex', vector.real.astype(dtype=np.complex)[:100])


def get_fourier(matrix: np.ndarray):
    show_img(f'input', matrix)

    print(f'im_result: {matrix.shape}')
    result = np.fft.fft(np.array(matrix))
    print('matrix', matrix)
    print('result', result)
    iFFT_complex = np.fft.ifft(result)
    ifft_real = iFFT_complex.real.astype(dtype=np.uint8)
    print(type(result[0]))
    vector_smoothing(ifft_real[0])

    show_img(f'refresh', ifft_real)
    show_img(f'one_line', np.array(ifft_real[0]))
    len_vector = len(result[0])
    # plt.scatter(range(len_vector), result[0])
    # print(result)
    # print(result.shape)
    # plt.plot(result.T, color='orange')
    # plt.show()


def main():
    im = Image.open('samples/file_example_TIFF_1MB.tiff')
    #im = Image.open('Screenshot from 2022-11-09 23-42-57.png')
    #im = Image.open('2021-04-05 19:38:33.9608011_1.jpg')
    #im.show()
    imarray = np.array(im)

    im2 = ImageOps.grayscale(Image.fromarray(imarray))
    print('im2', )

    images_blur = []
    for i in range(1, 7):
        gausian = 10 + i
        if gausian == 1:
            continue
        temp_blure = np.array(im2.filter(ImageFilter.GaussianBlur(gausian)))
        images_blur.append(temp_blure)
        show_img(f'{gausian} blur', temp_blure)

    im_result = images_blur[0].copy()
    show_img(f'{0} concatenate', im_result)

    print('type im_result', im_result)
    for num, img_b in enumerate(images_blur[1:]):
        im_result += img_b
        show_img(f'{num + 1} concatenate', im_result)

    get_fourier(im_result)

    cv.waitKey(100000)


if __name__ == '__main__':

    main()
