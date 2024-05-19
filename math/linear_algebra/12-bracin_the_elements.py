#!/usr/bin/env python3

"""
This script performs mathematical operations

"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division.

    Args:
        mat1 (numpy.ndarray): First input matrix.
        mat2 (numpy.ndarray): Second input matrix.

    Returns:
        tuple: Tuple containing the element-wise
        sum, difference, product, and quotient.
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
