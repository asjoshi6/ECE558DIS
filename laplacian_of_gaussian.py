"""
Possible implementations:
1) Convolve image with Gaussian, then take the difference of Gaussian - Approximation to LoG
2) Direct scipy implementation
3) Opencv implementation -> Blur first, then laplacian
"""

import cv2
import os
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import conv2d, normalize_image, dft2d_fft_based, idft2d_fft_based, pad_image
from helper_functions import maxima


def generate_log_apprximation(standard_deviation):
    # kernel_size = (standard_deviation * 4) + 1
    kernel_size = np.ceil(standard_deviation * 6)
    location_values = np.arange(int(-kernel_size//2), int(kernel_size//2 + 1))
    gaussian_1d = np.exp(-(location_values ** 2)/(2 * (standard_deviation ** 2)))
    log2d = (-1/(np.pi * (standard_deviation ** 4))) * (1 - (((location_values[np.newaxis, :] ** 2) +
             location_values[:, np.newaxis] ** 2)/(2 * (standard_deviation ** 2)))) *\
             (gaussian_1d[np.newaxis, :] * gaussian_1d[:, np.newaxis])
    return log2d


def generate_log_stack(img, number_of_iterations, init_sigma, k):
    stacked_logs = []
    img = img / 255
    for i in range(number_of_iterations):
        current_sigma = init_sigma * (k ** i)
        log_approx = generate_log_apprximation(current_sigma) * (current_sigma ** 2)
        filtered = conv2d(img, log_approx)
        # img_freq = dft2d_fft_based(img)
        # if (img.shape[1] - log_approx.shape[1]) % 2 == 1:
        #     log_approx
        # if (img.shape[0] - log_approx.shape[0]) % 2 == 0:
        #     pad_height = int((img.shape[0] - log_approx.shape[0]) // 2)
        # else:
        #     pad_height = int((img.shape[0] - log_approx.shape[0]) // 2) + 1
        # padded_log_approx = pad_image(log_approx, pad_width, pad_height)
        # padded_log_approx_freq = dft2d_fft_based(padded_log_approx)
        # filtered_freq = np.multiply(padded_log_approx_freq, img_freq)
        # filtered = idft2d_fft_based(filtered_freq)
        filtered = np.square(filtered)
        stacked_logs.append(filtered)
    return stacked_logs


def find_initial_keypoints(log_stack, threshold):
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


if __name__ == "__main1__":
    img_path = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\TestImages4Project3-Option3\TestImages4Project\butterfly.jpg"
    save_folder = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\delete\delete1"
    img = cv2.imread(img_path, 0)
    img_rgb = cv2.imread(img_path)
    sigma = 2
    number_of_iter = 8
    threshold = 0.09
    K = np.sqrt(2)
    stacked_logs = generate_log_stack(img, number_of_iter, sigma, K)
    for i, log in enumerate(stacked_logs):
        plt.imshow(log, cmap="gray")
        plt.savefig(os.path.join(save_folder, "scale_%d_log.png" % i))
    initial_keypoints = find_initial_keypoints(np.asarray(stacked_logs), threshold)
    maxima_locations = np.where(initial_keypoints == 1)
    for z, x, y in zip(maxima_locations[0], maxima_locations[1], maxima_locations[2]):
        radius = sigma * (K ** z)
        cv2.circle(img_rgb, (y, x), int(radius), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(save_folder, "key_points.png"), img_rgb)
    print("")

