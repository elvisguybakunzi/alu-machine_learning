#!/usr/bin/env python3
"""Module to perform random image cropping"""

import tensorflow as tf


def crop_image(image, size):
    """
    Function that performs a random crop of an image
    
    Args:
        image: 3D tf.Tensor containing the image to crop
        size: tuple containing the size of the crop
        
    Returns:
        tf.Tensor: The randomly cropped image
    """
    return tf.image.random_crop(image, size)