#!/usr/bin/env python3
"""The scirpt that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    args:
    image (np.ndarray): shape(m, h, w) containing multiple grayscale images.
    kernel (np.ndarray): shape(kh, kw) containing the kernel for convolution.

    Returns:
      np.ndarray: The convolved images.

    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the dimension of the output
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initilaize the output array
    output = np.zeros((m, output_h, output_w))

    # Perform the convolution
    for i in range(output_h):
        for j in range(output_w):
            # Perform element-wise multiplication and sum the reslut
            output[:, i, j] = np.sum(
              images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    return output
