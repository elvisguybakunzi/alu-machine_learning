#!/usr/bin/env python3
"""Script that performs a same convolution
    on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images (numpy.ndarray): (m, h, w) containing multiple
    grayscale images
      - m is the number of images
      - h is the height in pixels og the images
      - w is the width in pixels of the images
    - kernel (numpy.ndarray): (kh, kw) containing for
    the convolution
      - kh is the height of the kernel
      - kw is the width of the kernel

    Returns:
    - numpy.ndarray: containing the convolved images

    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(
       images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant'
       )

    # Initialize the output with zeros
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            # Prform element-wise multiplication and sum the result
            output[:, i, j] = np.sum(
              padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return output
