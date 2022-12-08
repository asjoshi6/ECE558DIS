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
