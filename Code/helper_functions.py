import numpy as np


def dft2d_fft_based(image):
    """
    Implementation of DFT2D using the 1D FFT using the separable property of DFT2D
    :param image: The image for which the DFT2D transformation needs to be computed
    :return: The frequency domain transformed image, spectrum and corresponding phase
    """
    image = np.asarray(image, np.complex)
    for i in range(image.shape[0]):
        image[i, :] = np.fft.fft(image[i, :])
    for i in range(image.shape[1]):
        image[:, i] = np.fft.fft(image[:, i])
    return image


def idft2d_fft_based(image):
    """
    Implementation of IDFT2D using the 1D FFT
    :param image: The image for which the DFT2D transformation needs to be computed
    :return: The frequency domain transformed image, spectrum and corresponding phase
    """
    for i in range(image.shape[0]):
        image[i, :] = np.array(1j, np.complex) * np.conjugate(image[i, :])
        image[i, :] = np.fft.fft(image[i, :])
        image[i, :] = np.array(1j, np.complex) * np.conjugate(image[i, :])
    image /= image.shape[0]
    for i in range(image.shape[1]):
        image[:, i] = np.array(1j, np.complex) * np.conjugate(image[:, i])
        image[:, i] = np.fft.fft(image[:, i])
        image[:, i] = np.array(1j, np.complex) * np.conjugate(image[:, i])
    image /= image.shape[1]
    return image
