import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter


class GaussianFilter:
    def __init__(self, image):
        self.origin_image: np.ndarray = image.copy()
        self.kernel_size = 6, 6
        self.smoothing_images = []

    def blur(self, kernel=None, is_pil=True):
        # if kernel is not None:
        self.kernel_size = kernel, kernel
        print(f'using kernel: {self.kernel_size}')
        if is_pil:
            blur_images = np.array(Image.fromarray(self.origin_image).filter(ImageFilter.GaussianBlur(kernel)))
        else:
            blur_images = cv.GaussianBlur(self.origin_image, self.kernel_size, 0)
        self.smoothing_images.append(blur_images)
        return self.smoothing_images[-1]

    def get_concatenate_blur_images(self):
        result = self.smoothing_images[0]
        for smoth_img in self.smoothing_images[1:]:
            result += smoth_img
        return result
