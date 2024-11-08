#!/usr/bin/env python3
"""Module to perform image flipping"""

import tensorflow as tf


def flip_image(image):
    """
    Function that flips an image horizontally
    
    Args:
        image: 3D tf.Tensor containing the image to flip
        
    Returns:
        tf.Tensor: The flipped image
    """
    return tf.image.flip_left_right(image)