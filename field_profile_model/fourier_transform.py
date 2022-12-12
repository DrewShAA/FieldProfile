import numpy as np


class FourierTransformator:
    def __init__(self, image: np.ndarray):
        print(f'shape_images: {image.shape}')
        self.origin_image: np.ndarray = image.copy()
        self.frequency_spectrum_complex: np.ndarray = ...
        self.frequency_spectrum_real: np.ndarray = ...

    def get_fourier_complex(self) -> np.ndarray:
        if self.frequency_spectrum_complex is not ...:
            return self.frequency_spectrum_complex

        self.frequency_spectrum_complex = np.fft.fft(self.origin_image)
        return self.frequency_spectrum_complex

    def get_fourier_real(self) -> np.ndarray:
        if self.frequency_spectrum_real is not ...:
            return self.frequency_spectrum_real

        # if self.frequency_spectrum_complex is not ...:
        #     self.frequency_spectrum_real =\
        #         self.frequency_spectrum_complex.real.astype(dtype=np.uint8)
        #     return self.frequency_spectrum_real

        self.frequency_spectrum_real =\
            np.fft.rfft(self.origin_image)#.real.astype(dtype=np.uint8)
        return self.frequency_spectrum_real
