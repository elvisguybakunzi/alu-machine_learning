#!/usr/bin/env python3
"""Script that performs convolution with padding and stride"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with specified
    padding and stride.

    Parameters:
    - images (numpy.ndarray): (m, h, w) containing multiple grayscale images
      - m is the number of images
      - h is the height in pixels of the images
      - w is the width in pixels of the images
    - kernel (numpy.ndarray): (kh, kw) containing the kernel for
    the convolution
      - kh is the height of the kernel
      - kw is the width of the kernel
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
      - if 'same', performs a same convolution
      - if 'valid', performs a valid convolution
      - if a tuple: (ph, pw) padding for the height and width of the image
    - stride (tuple): (sh, sw)
      - sh is the stride for the height of the image
      - sw is the stride for the width of the image

    Returns:
    - numpy.ndarray: containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Pad the images with zeros
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
        )

    # Calculate the dimensions of the output
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output with zeros
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Calculate the start and end positions for the current slice
            i_start = i * sh
            i_end = i_start + kh
            j_start = j * sw
            j_end = j_start + kw

            # Perform element-wise multiplication and sum the results
            output[:, i, j] = np.sum(
                padded_images[:, i_start:i_end, j_start:j_end] * kernel, axis=(1, 2)
            )

    return output
