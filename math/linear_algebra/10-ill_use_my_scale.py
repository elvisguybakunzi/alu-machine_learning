#!/usr/bin/env python3


def np_shape(matrix):
    """Calculate the shape of a nested list simulating a numpy.ndarray."""
    def shape_rec(mat, shape=[]):
        shape.append(len(mat))
        if len(mat) > 0 and isinstance(mat[0], list):
            return shape_rec(mat[0], shape)
        return shape
    
    return tuple(shape_rec(matrix))
