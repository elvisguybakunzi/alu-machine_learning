#!/usr/bin/env python3

"""
This script calculates the shape of numpy.ndarray.

"""


import numpy as np


def np_shape(matrix):
    """Calculate the shape of a nested list."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return tuple(shape)
