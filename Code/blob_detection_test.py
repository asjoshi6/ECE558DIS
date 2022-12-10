import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from blob_detector_functions import generate_log_stack, find_initial_keypoints, non_maxima_suppression


def run_blob_detection(img_path, sigma=1.5, save_folder=None, number_of_iterations=8, threshold=0.02, k=1.5, viz=True):
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


if __name__ == "__main__":
    image_path = "../test_images/butterfly.jpg"
    save_folder_path = "../results"
    sigma_val = 1.5
    number_of_iter = 8
    maxima_threshold = 0.02
    K = 1.5
    visualize_results = True
    start = time.time()
    run_blob_detection(image_path, sigma_val, save_folder_path, number_of_iter, maxima_threshold, K, visualize_results)
    print("Time taken for blob detection: ", time.time() - start)
