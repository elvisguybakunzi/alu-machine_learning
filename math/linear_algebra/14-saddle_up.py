#!/usr/bin/env python3

"""
This script perform matrix multiplication.

"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication.

    Args:
        mat1 (numpy.ndarray): First input matrix.
        mat2 (numpy.ndarray): Second input matrix.

    Returns:
        numpy.ndarray: Result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
