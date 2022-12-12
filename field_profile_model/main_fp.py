import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())

from field_profile_model import gaussian
from field_profile_model import experimental_smoothing as exp_smt
from field_profile_model.fourier_transform import FourierTransformator
from field_profile_model import visualization_plt


def main():
    img = cv.imread('samples/file_example_TIFF_1MB.tiff')
    img = cv.imread('samples/N047E039/ALPSMLC30_N047E039_STK.tif')
    img = cv.imread('/home/drew/Projects/field_profile/samples/N047E039/ALPSMLC30_N047E039_STK.tif')
    img = cv.imread('/home/drew/Projects/field_profile/samples/example_1/example_1.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)


    visualization_plt.show_img('img_gray', img_gray)
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 100  # Brightness control (0-100)
    bright_result = cv.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

    print(bright_result.shape)
    visualization_plt.show_img('brightness', bright_result[-2000:, :2000])

    res = exp_smt.get_border_by_prewiit(bright_result)
    visualization_plt.show_img('prewiit_result', res)

    # fourier_bright = FourierTransformator(manual_result)
    # frequency_o_complex_bright = fourier_bright.get_fourier_complex()
    #
    # gauss = gaussian.GaussianFilter(img_gray)
    # img_3 = gauss.blur(3)
    # img_10 = gauss.blur(9)
    # img_10 = gauss.blur(7)
    # img_10 = gauss.blur(5)
    # img_10 = gauss.blur(21)
    # img_10 = gauss.blur(111)
    # img_10 = gauss.blur(223)
    # img_1 = gauss.blur(1)
    #img_1 = gauss.blur(2)
    # new_img = gauss.get_concatenate_blur_images()
    # img_10_manual_result = cv.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    # visualization_plt.show_img('gauss', new_img)
    # res_2 = exp_smt.get_border_by_prewiit(new_img)
    # visualization_plt.show_img('img_res2', res_2)


    fourier = FourierTransformator(img_gray)
    frequency_o_real = fourier.get_fourier_real()

    # before
    print(frequency_o_real[0])
    visualization_plt.visualization_vector(frequency_o_real[0])

    # after fourier
    result_line = exp_smt.flf(frequency_o_real[0])
    print(result_line)
    # visualization_plt.visualization_vector(result_line)

    #
    # visualization_plt.visualization_vector(result_line)
    print('input', frequency_o_real)

    plt.scatter(range(len(frequency_o_real[0])), frequency_o_real[0])
    # plt.scatter(range(len(frequency_o_complex_bright[0])), frequency_o_complex_bright[0])
    #plt.plot(vector, color='orange')
    plt.show()

    result = exp_smt.flf_matrix(frequency_o_real)
    print('output', result)
    ifft_complex = np.fft.irfft(result)
    ifft_real = ifft_complex#.real.astype(dtype=np.uint8)
    print(ifft_real)
    visualization_plt.show_img('img_gray', img_gray)
    visualization_plt.show_img('восстановленное', ifft_real)

    # key = 1
    # while key != ord('q'):
    cv.waitKey(0)
    # frequency_o_real = fourier.get_fourier_real()


    # real = frequency_o_complex.real.astype(dtype=np.uint8)
    # print(real)
    # cv.imwrite('charts/test1', real)


if __name__ == "__main__":
    main()
