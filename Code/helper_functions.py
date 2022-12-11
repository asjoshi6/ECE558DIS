import numpy as np


def dft2d_fft_based(image):
    """
    Implementation of DFT2D using the 1D FFT using the separable property of DFT2D
    :param image: The image for which the DFT2D transformation needs to be computed
            type: ndarray
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
            type: complex ndarray
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


def pad_kernel(kernel, target_size):
    """
    Zero Pads the given kernel to the target size
    :param kernel: A ndarray which needs to be padded
           type: ndarray
    :param target_size: Size to which the kernel needs to be padded
           type: tuple/list
    :return: Padded kernel (ndarray)
    """
    padded_kernel = np.zeros(target_size)
    x_position_1 = int(max(0, int((padded_kernel.shape[0] - kernel.shape[0]) / 2)))
    x_position_2 = int(min(target_size[0], ((padded_kernel.shape[0] + kernel.shape[0]) / 2)))
    y_position_1 = int(max(0, int((padded_kernel.shape[1] - kernel.shape[0]) / 2)))
    y_position_2 = int(min(target_size[1], ((padded_kernel.shape[1] + kernel.shape[0]) / 2)))
    padded_kernel[x_position_1 : x_position_2, y_position_1: y_position_2] = kernel
    return padded_kernel
