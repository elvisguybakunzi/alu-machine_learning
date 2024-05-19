#!/usr/bin/env python3

"""
This script calculates the shape of numpy.ndarray.

"""


def np_shape(matrix):
    """Calculate the shape of a numpy.ndarray represented as nested lists."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return tuple(shape)
