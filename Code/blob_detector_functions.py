"""
Possible implementations: Compare correctness and speed
1) Convolve image with Gaussian, then take the difference of Gaussian - Approximation to LoG
2) Direct scipy implementation
3) Opencv implementation -> Blur first, then laplacian
4) Directly compute DoG from formula
"""

import numpy as np
from helper_functions import dft2d_fft_based, idft2d_fft_based, pad_kernel


def generate_log_apprximation(standard_deviation):
    """
    Computes the approximation of Laplacian of Gaussian using Difference of Gaussian given the standard deviation
    for the Gaussian kernel
    :param standard_deviation: The standard deviation for the Gaussian kernel/filter which is to be used for computed
                               the Difference of Gaussian
           type: float
    :return: 2D Laplacian of Gaussian kernel
    """
    # kernel_size = (standard_deviation * 4) + 1
    kernel_size = np.ceil(standard_deviation * 6)
    location_values = np.arange(int(-kernel_size//2), int(kernel_size//2 + 1))
    gaussian_1d = np.exp(-(location_values ** 2)/(2 * (standard_deviation ** 2)))
    log2d = (-1/(np.pi * (standard_deviation ** 4))) * (1 - (((location_values[np.newaxis, :] ** 2)
             + location_values[:, np.newaxis] ** 2)/(2 * (standard_deviation ** 2)))) *\
             (gaussian_1d[np.newaxis, :] * gaussian_1d[:, np.newaxis])
    return log2d


def generate_log_stack(img, number_of_iterations, init_sigma, k):
    """
    Computes the scale space/LoG stack for the given image
    :param img: The image for which the LoG stack needs to be computed
          type: ndarray
    :param number_of_iterations: The number of levels in the LoG space
           type: int
    :param init_sigma: The standard deviation for the first level of the LoG
           type: float
    :param k: The factor by which the standard deviation is scaled in consequent levels in LoG scale space
              (sigma, k*sigma, k^2*(sigma), ..)
           type: float
    :return: Stack of LoG
    """
    stacked_logs = []
    img = img / 255
    for i in range(number_of_iterations):
        current_sigma = init_sigma * (k ** i)
        log_approx = generate_log_apprximation(current_sigma) * (current_sigma ** 2)
        padded_log_approx = pad_kernel(log_approx, img.shape)
        padded_log_approx = np.fft.ifftshift(padded_log_approx)
        img_freq = dft2d_fft_based(img)
        padded_log_approx_freq = dft2d_fft_based(padded_log_approx)
        filtered_freq = np.multiply(padded_log_approx_freq, img_freq)
        filtered = idft2d_fft_based(filtered_freq)
        filtered = np.abs(filtered)
        filtered = np.square(filtered)
        stacked_logs.append(filtered)
    return stacked_logs


def find_initial_keypoints(log_stack, threshold):
    """
    Computes the initial key points from the stacked LoG by comparing each value with its 3D neighborhood,
    picking the maximum value and checking if its value is greater than the given threshold
    :param log_stack: The stack of LoG responses of the image
           type: ndarray
    :param threshold: The threshold used for selecting maximas in the LoG
           type: float
    :return: A binary mask with the detected keypoints/maximas being 1
    """
    number_of_logs, height, width = log_stack.shape
    maxima_mask = np.zeros_like(log_stack)
    for i in range(1, number_of_logs - 1):
        for j in range(1, height - 1):
            for k in range(1, width - 1):
                comparison_cube = log_stack[i - 1: i + 2, j - 1: j + 2, k - 1: k + 2]
                cube_max_value_index = np.unravel_index(np.argmax(comparison_cube), comparison_cube.shape)
                if comparison_cube[cube_max_value_index] > threshold:
                    maxima_mask[i - 1 + cube_max_value_index[0], j - 1 + cube_max_value_index[1], k - 1 + cube_max_value_index[2]] = 1
    return maxima_mask


def non_maxima_suppression(initial_maxima_mask, log_stack):
    """
    Performs Non Maxima suppression in 2D neighbourhood of each maxima and then in the 3D neighborhood of each maxima
    and maintains only the key points which are a maxima in both the 2D and 3D neighborhoods
    :param initial_maxima_mask: A binary mask indicating the initially computed maxima values
           type: ndarray
    :param log_stack: Stack of the LoG responses
           type: ndarray
    :return: A binary mask indicating the filtered key points/maximas
    """
    number_of_scales = initial_maxima_mask.shape[0]
    local_maximas_2d = np.zeros_like(log_stack)
    for i in range(number_of_scales):
        maxima_locations = np.where(initial_maxima_mask[i] == 1)
        for x_i, y_i in zip(maxima_locations[0], maxima_locations[1]):
            local_neighborhood = log_stack[i, max(0, x_i - 1): x_i + 2, max(y_i - 1, 0): y_i + 2]
            local_maximas_2d[i, x_i, y_i] = np.max(local_neighborhood)

    local_maxima_3d = np.zeros_like(log_stack)
    for i in range(1, number_of_scales - 1):
        maxima_locations = np.where(initial_maxima_mask[i] == 1)
        for x_i, y_i in zip(maxima_locations[0], maxima_locations[1]):
            local_neighborhood_3d = local_maximas_2d[i - 1: i + 2, max(0, x_i - 1): x_i + 2, max(0, y_i - 1): y_i + 2]
            local_maxima_3d[i, x_i, y_i] = np.max(local_neighborhood_3d)

    final_maxima_locations = (local_maxima_3d == local_maximas_2d) * initial_maxima_mask
    return final_maxima_locations

