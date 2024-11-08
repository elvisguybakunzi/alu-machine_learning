#!/usr/bin/env python3
"""Module to perform random image shearing"""

import tensorflow as tf


def shear_image(image, intensity):
    """
    Function that randomly shears an image
    
    Args:
        image: 3D tf.Tensor containing the image to shear
        intensity: intensity with which the image should be sheared
        
    Returns:
        tf.Tensor: The sheared image
    """
    # Convert intensity to radians (normalize and scale)
    shear = intensity / 100  # Convert to smaller scale
    
    # Create random shear angle between -shear and shear
    shear_angle = tf.random.uniform([], -shear, shear)
    
    # Create the shear transformation matrix
    shear_matrix = [1.0, shear_angle, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]
    shear_matrix = tf.reshape(shear_matrix, [3, 3])
    
    # Get image shape
    shape = tf.cast(tf.shape(image), dtype=tf.float32)
    
    # Create transformation that centers the image at origin
    height, width = shape[0], shape[1]
    tx = width / 2
    ty = height / 2
    
    translation_matrix = tf.convert_to_tensor([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    
    inverse_translation_matrix = tf.convert_to_tensor([
        [1.0, 0.0, -tx],
        [0.0, 1.0, -ty],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    
    # Combine transformations
    transform = tf.matmul(translation_matrix,
                         tf.matmul(shear_matrix, inverse_translation_matrix))
    
    # Apply transformation
    sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.cast(shape[:2], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT")
    
    return tf.squeeze(sheared_image)