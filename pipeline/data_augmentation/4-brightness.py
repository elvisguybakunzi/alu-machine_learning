#!/usr/bin/env python3
"""Module to perform random brightness adjustment"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Function that randomly changes the brightness of an image
    
    Args:
        image: 3D tf.Tensor containing the image to change
        max_delta: maximum amount the image should be brightened (or darkened)
        
    Returns:
        tf.Tensor: The brightness-adjusted image
    """
    return tf.image.random_brightness(image, max_delta)