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
from helper_functions import conv2d


def generate_gaussian_filter(standard_deviation):
    kernel_size = (standard_deviation * 4) + 1
    # kernel_size = np.ceil(standard_deviation * 6)
    location_values = np.arange(int(-kernel_size//2), int(kernel_size//2 + 1))
    gaussian_1d = np.exp(-(location_values ** 2)/(2 * (standard_deviation ** 2)))
    gaussian2d = (gaussian_1d[np.newaxis, :] * gaussian_1d[:, np.newaxis]) / (2 * np.pi * (standard_deviation ** 2))
    return gaussian2d


def scale_space_generation(image, initial_sigma=np.sqrt(2)/2, number_of_scales=5, number_of_octaves=4):
    log_stack = []
    sigma = initial_sigma
    k = 2 ** (1/(number_of_scales - 3))
    log_stack_per_octave = []
    start = 0
    for i in range(number_of_octaves):
        blurred_image_scale_j_1 = None
        if len(log_stack_per_octave):
            start = 1
        for j in range(start, number_of_scales):
            current_sigma = sigma * (k ** j)
            gaussian_kernel = generate_gaussian_filter(current_sigma)
            blurred_image_scale_j = conv2d(image, gaussian_kernel)
            # blurred_image_scale_j = cv2.filter2D(image, -1, gaussian_kernel)
            # blurred_image_scale_j = np.square(blurred_image_scale_j)
            # blurred_image_scale_j = np.asarray(blurred_image_scale_j, np.int32)
            if blurred_image_scale_j_1 is not None:
                # log_stack_per_octave.append((blurred_image_scale_j - blurred_image_scale_j_1) ** 2)
                log_stack_per_octave.append(np.asarray((blurred_image_scale_j - blurred_image_scale_j_1) ** 2, np.uint8))
            blurred_image_scale_j_1 = blurred_image_scale_j
        # height, width = img.shape
        # image = cv2.resize(image, (int(width//2), int(height//2)))
        sigma *= 2
        log_stack.append(log_stack_per_octave)
        log_stack_per_octave = [log_stack_per_octave[-3]]
    return log_stack


if __name__ == "__main1__":
    img_path = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\TestImages4Project3-Option3\TestImages4Project\butterfly.jpg"
    save_folder = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\delete"
    scales = [1, 3, 5, 7, 9]
    peaks_list = []
    for gaussian_scale in scales:
        save_path = os.path.join(save_folder, "scipy_kernel_%d.png" % gaussian_scale)
        save_path_2 = os.path.join(save_folder, "scipy_kernel_%d_normalized.png" % gaussian_scale)
        img = cv2.imread(img_path, 0)
        log = gaussian_laplace(img, gaussian_scale)
        log_normalized = (gaussian_scale**2) * log
        cv2.imwrite(save_path, log)
        cv2.imwrite(save_path_2, log_normalized)
    print("")


if __name__ == "__main__":
    img_path = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\TestImages4Project3-Option3\TestImages4Project\butterfly.jpg"
    save_folder = r"C:\Users\sumuk\OneDrive\Desktop\NCSU_related\Courses_and_stuff\Courses_and_stuff\NCSU_courses_and_books\ECE_558\Project03\delete\delete1"
    img = cv2.imread(img_path, 0)
    log_stack = scale_space_generation(img)
    for i, octave in enumerate(log_stack):
        for j, dog in enumerate(octave):
            plt.figure()
            plt.imshow(dog, cmap="gray")
            save_path = os.path.join(save_folder, "octave_%d_scale_%d_dog.png" % (i, j))
            plt.savefig(save_path)
    print("")
