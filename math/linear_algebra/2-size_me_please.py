#!/usr/bin/env python3
"""
This Python script is used to calculate shape of matrix.

"""


def matrix_shape(matrix):

    """
    Calculates the shape of a matrix.

    Parameters:
    matrix (list): A list of lists (or nested lists) representing the matrix.

    Returns:
    list: A list of integers representing the dimensions of the matrix.

    """
    if not isinstance(matrix, list):
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
