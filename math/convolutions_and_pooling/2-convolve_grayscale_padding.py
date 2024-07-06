#!/usr/bin/env python3
"""Script that performs convolution with custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): A numpy.ndarray with shape (m, h, w)
        containing multiple grayscale images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
        kernel (numpy.ndarray): A numpy.ndarray with shape (kh, kw)
        containing the kernel for the convolution.
            kh is the height of the kernel.
            kw is the width of the kernel.
        padding (tuple): A tuple of (ph, pw) where:
            ph is the padding for the height of the image.
            pw is the padding for the width of the image.

    Returns:
        numpy.ndarray: A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
        )

    # Perform convolution
    conv_h = h + 2 * ph - kh + 1
    conv_w = w + 2 * pw - kw + 1
    conv_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            # Prform element-wise multiplication and sum the result
            conv_images[:, i, j] = np.sum(
              padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return conv_images
