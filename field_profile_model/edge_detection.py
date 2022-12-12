import cv2
import numpy as np
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())

from field_profile_model import visualization_plt

structs = {
    'cross3': np.array([[False, True, False],
                        [True,  True, True],
                        [False, True, False]], dtype=bool),
    'rect3': np.ones((5,5), dtype=bool),
    'cross5': np.array([
        [False,False,True,False,False],
        [False,False,True,False,False],
        [True]*5,
        [False,False,True,False,False],
        [False,False,True,False,False]], dtype=bool),
    'rect5': np.ones((5,5), dtype=bool)
}


def binaring(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = 0 if matrix[i][j] < 20 else 255
    return matrix


def dilate(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    y = kernal.shape[0] // 2
    x = kernal.shape[1] // 2
    processed_image = np.copy(img)
    for i in range(y, img.shape[0] - y):
        for j in range(x, img.shape[1] - x):
            local_window = img[i-y:i+y+1, j-x:j+x+1]
            processed_image[i][j] = np.max(local_window[kernal])
    return processed_image


def erosion(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    y = kernal.shape[0] // 2
    x = kernal.shape[1] // 2
    processed_image = np.copy(img)
    for i in range(y, img.shape[0] - y):
        for j in range(x, img.shape[1] - x):
            local_window = img[i-y:i+y+1, j-x:j+x+1]
            processed_image[i][j] = np.min(local_window[kernal])
    return processed_image


imge = cv2.imread('/home/drew/Projects/field_profile/samples/example_1/example_1.png')

print(imge.shape)
# imge = cv2.imread('samples/2022-10-31_00_31_54.806920_sec_2.jpg')
# imge = cv2.imread('samples/file_example_TIFF_1MB.tiff')

img = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)#[-2000:, :2000]
alpha = 2  # Contrast control (1.0-3.0)
beta = 100  # Brightness control (0-100)
# img = cv2.convertScaleAbs(imge, alpha=alpha, beta=beta)

#gaussian
img_gaussian = cv2.GaussianBlur(img, (1, 1), 0)
for i in range(11, 22, 2):
    temp_blur = cv2.GaussianBlur(img, (i, i), 0)
    img_gaussian += temp_blur
    visualization_plt.show_img(f"{i} Gaussian", temp_blur)
#img_gaussian = cv2.convertScaleAbs(img_gaussian, alpha=alpha, beta=beta)

#canny
img_canny = cv2.Canny(img_gaussian, 100, 300)
# img_canny_dilate = dilate(img_canny, structs['rect3'])
# img_canny_erosion = erosion(img_canny_dilate, structs['rect3'])


#sobel
sobel_kernal = 5
img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=sobel_kernal)
img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=sobel_kernal)
img_sobel = img_sobelx + img_sobely

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

#laplacian
laplacian_img = cv2.Laplacian(img_gaussian, 5, cv2.CV_8U)
# img_prewitt_dilate = dilate(img_prewitt, structs['rect3'])

# img_prewitt = binaring(img_prewitt)

visualization_plt.show_img("Original Image", imge)
visualization_plt.show_img("scaleABS Image", img)
visualization_plt.show_img("Canny", img_canny)
visualization_plt.show_img("Sobel X", img_sobelx)
visualization_plt.show_img("Sobel Y", img_sobely)
visualization_plt.show_img("Sobel", img_sobel)
visualization_plt.show_img("Prewitt X", img_prewittx)
visualization_plt.show_img("Prewitt Y", img_prewitty)
visualization_plt.show_img("Prewitt", img_prewitt)
# visualization_plt.show_img("Canny dilate", img_canny_dilate)
# visualization_plt.show_img("Canny erosion", img_canny_erosion)
visualization_plt.show_img("img_gaussian", img_gaussian)
visualization_plt.show_img("laplacian", laplacian_img)
# cv2.imshow("Original Image", img)
# cv2.imshow("Canny", img_canny)
# cv2.imshow("Sobel X", img_sobelx)
# cv2.imshow("Sobel Y", img_sobely)
# cv2.imshow("Sobel", img_sobel)
# cv2.imshow("Prewitt X", img_prewittx)
# cv2.imshow("Prewitt Y", img_prewitty)
# cv2.imshow("Prewitt", img_prewittx + img_prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()