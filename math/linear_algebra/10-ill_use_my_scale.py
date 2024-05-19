#!/usr/bin/env python3


def np_shape(matrix, depth=0):
    """Calculate the shape of a numpy.ndarray."""
    return (not isinstance(matrix, list) and () or
            (depth == 0 and (len(matrix),) or
             (len(matrix),) + np_shape(matrix[0], depth + 1)))
