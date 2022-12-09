import numpy as np


def pad_image(image, padding_width, padding_height):
    """
    Function to perform the padding to the image given the padding parameters
    :param image: The image for which the padding needs to be performed
           type: A 2D single or multichannel array
    :param padding_width: The number of columns which needs to be added for padding on each side
                    type: int
    :param padding_height: The number of rows which needs to be added for padding on each side
                     type: int
    :return: returns the padded image
    """
    padded_image = np.zeros(
        (image.shape[0] + (2 * padding_height), image.shape[1] + (2 * padding_width), image.shape[-1]))
    padded_image[padding_height: padded_image.shape[0] - padding_height,
                 padding_width: padded_image.shape[1] - padding_width, :] = image
    return padded_image


def conv2d(f, w):
    """
    Computes the cross correlation between the input image f and the provided filter w
    :param f: The input image which can be either a gray or color image
           type: A 2D single or multichannel array
    :param w: The provided kernel with which the convolution needs to be performed
           type: A 2D ndarray
    :param pad: Indicates the padding type that needs to be performed for the image
           type: string
    :param stride: Indicates the stride of the kernel when convolving with the image
           type: int
    :return: Returns the convolved image
    """
    input_width = f.shape[1]
    input_height = f.shape[0]
    if len(f.shape) == 2:
        f = f[:, :, np.newaxis]
    input_depth = f.shape[-1]
    kernel_width = w.shape[1]
    kernel_height = w.shape[0]
    padded_image = pad_image(f, kernel_width // 2, kernel_height // 2)
    output_width = int((input_width - kernel_width + (2 * (kernel_width // 2))) + 1)
    output_height = int((input_height - kernel_height + (2 * (kernel_height // 2))) + 1)
    output_image = np.zeros((output_height, output_width, input_depth))
    w = np.flip(w)
    for i in range(0, input_width):
        for j in range(0, input_height):
            for k in range(input_depth):
                output_image[j, i, k] = np.sum(padded_image[j: j + kernel_height, i: i + kernel_width, k] * w)
    output_image_cropped = np.squeeze(output_image[: input_height, : input_width, :])
    return output_image_cropped


def normalize_image(upper_limit, img):
    return np.asarray(((img - np.min(img)) / (np.max(img) - np.min(img))) * upper_limit, np.int8)


def maxima(logstack, k, sigma, threshold):
    maxlocation = []
    row, col = logstack[0].shape
    for i in range(row):
        for j in range(col):
            ##annotation of box i.e neigbhourhood region
            box = logstack[:, i:i + 2, j:j + 2]
            extrema = np.max(box)
            if extrema >= threshold:
                z, x, y = np.unravel_index(box.argmax(), box.shape)
                maxlocation.append((x + i, y + j, np.power(k, z) * sigma))
    return maxlocation


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
