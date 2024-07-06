#!/usr/bin/env python3
"""Script that performs a convolution on images with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters:
    - images (numpy.ndarray): (m, h, w, c) containing multiple images
      - m is the number of images
      - h is the height in pixels of the images
      - w is the width in pixels of the images
      - c is the number of channels in the image
    - kernel (numpy.ndarray): (kh, kw, c) containing the kernel for
    the convolution
      - kh is the height of the kernel
      - kw is the width of the kernel
      - c is the number of channels in the kernel
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
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("The number of channels in the kernel "
                         "must match the number of channels in the image")

    # Calculate padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph = pw = 0
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        raise ValueError("Invalid padding value")

    # Pad the images with zeros
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
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
                padded_images[:, i_start:i_end, j_start:j_end, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
