import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from blob_detector_functions import generate_log_stack, find_initial_keypoints, non_maxima_suppression


def run_blob_detection(img_path, sigma=1.5, save_folder=None, number_of_iterations=8, threshold=0.02, k=1.5, viz=True):
    """
    Detects blobs for the image sorted at img_path with respect to the other parameters provided
    :param img_path: The path to the image for which the blob detection needs to be performed
           type: str
    :param sigma: The standard deviation for the first level in the scale space
           type: str
           default: 1.5
    :param save_folder: The path to the folder where the resulting visualizations need to be saved
           type: str
           default: None
    :param number_of_iterations: The number of scales/levels of the laplacian of Gaussian space
           type: int
           default: 8
    :param threshold: The threshold used to select the maxima values from the LoG response in a neighborhood fashion
           type: float
           default: 0.02
    :param k: The factor by which the standard deviation is scaled in consequent levels in LoG scale space
              (sigma, k*sigma, k^2*(sigma), ..)
           type: float
           default: 1.5
    :param viz: If the results need to be visualized from the LoG scale space, maxima detection and non maxima
                suppressed steps.
           type: bool
           default: True
    """
    img = cv2.imread(img_path, 0)
    stacked_logs = generate_log_stack(img, number_of_iterations, sigma, k)
    if viz:
        for i, log in enumerate(stacked_logs):
            plt.imshow(log, cmap="gray")
            plt.savefig(os.path.join(save_folder, "scale_%d_log.png" % i))
    initial_keypoints_mask = find_initial_keypoints(np.asarray(stacked_logs), threshold)
    initial_keypoints_locations = np.where(initial_keypoints_mask == 1)
    if viz:
        img_rgb = cv2.imread(img_path)
        for z, x, y in zip(initial_keypoints_locations[0], initial_keypoints_locations[1], initial_keypoints_locations[2]):
            radius = sigma * (K ** z)
            cv2.circle(img_rgb, (y, x), int(radius), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_folder, "key_points_initial.png"), img_rgb)
    if viz:
        img_rgb = cv2.imread(img_path)
        filtered_keypoints = non_maxima_suppression(initial_keypoints_mask, np.asarray(stacked_logs))
        maxima_locations = np.where(filtered_keypoints == 1)
        for z, x, y in zip(maxima_locations[0], maxima_locations[1], maxima_locations[2]):
            radius = sigma * (K ** z)
            cv2.circle(img_rgb, (y, x), int(radius), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_folder, "key_points.png"), img_rgb)


if __name__ == "__main1__":
    """
    Runtime Analysis
    """
    image_folder = "../test_images/"
    image_paths = list(os.listdir(image_folder))
    save_folder_path = "../results"
    sigma_val = [2, 1, 2, 2]
    number_of_iter = [8, 8, 8, 8]
    maxima_threshold = [0.015, 0.02, 0.02, 0.025]
    K = 1.5
    visualize_results = True
    total_time = 0
    for i, image_path in enumerate(image_paths):
        image_name = image_path.split(".")[0]
        save_folder_path_i = os.path.join(save_folder_path, image_name)
        if not os.path.exists(save_folder_path_i):
            os.mkdir(save_folder_path_i)
        start = time.time()
        run_blob_detection(os.path.join(image_folder, image_path), sigma_val[i], save_folder_path_i, number_of_iter[i],
                           maxima_threshold[i], K, visualize_results)
        total_time += (time.time() - start)
    print("Time taken for blob detection: ", total_time / len(image_paths))


if __name__ == "__main2__":
    """
    Blob detection for single image
    """
    image_path = "../test_images/butterfly.jpg"
    save_folder_path = "../results/butterfly"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    sigma_val = 2
    number_of_iter = 8
    maxima_threshold = 0.01
    K = 1.5
    visualize_results = True
    start = time.time()
    run_blob_detection(image_path, sigma_val, save_folder_path, number_of_iter,
                       maxima_threshold, K, visualize_results)
    print("Time taken for blob detection: ", (time.time() - start))

