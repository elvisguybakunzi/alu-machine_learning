#!/usr/bin/env python3
"""Script that performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    - images (numpy.ndarray): (m, h, w, c) containing multiple images
      - m is the number of images
      - h is the height in pixels of the images
      - w is the width in pixels of the images
      - c is the number of channels in the image
    - kernel_shape (tuple): (kh, kw) containing the kernel shape
    for the pooling
      - kh is the height of the kernel
      - kw is the width of the kernel
    - stride (tuple): (sh, sw)
      - sh is the stride for the height of the image
      - sw is the stride for the width of the image
    - mode (str): indicates the type of pooling
      - 'max' indicates max pooling
      - 'avg' indicates average pooling

    Returns:
    - numpy.ndarray: containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate the dimensions of the output
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize the output with zeros
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            # Calculate the start and end positions for the current slice
            i_start = i * sh
            i_end = i_start + kh
            j_start = j * sw
            j_end = j_start + kw

            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, i_start:i_end, j_start:j_end, :], axis=(1, 2)
                    )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, i_start:i_end, j_start:j_end, :], axis=(1, 2)
                    )
            else:
                raise ValueError("Invalid mode value")

    return output
