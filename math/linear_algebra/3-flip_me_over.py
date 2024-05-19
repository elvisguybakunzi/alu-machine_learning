#!/usr/bin/env python3

"""
This script Flip over the matrix

"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Parameters:
    matrix (list of lists): A 2D list representing the matrix.

    Returns:
    list of lists: A new 2D list representing the transposed matrix.
    """
    return [
        [matrix[row][col] for row in range(len(matrix))]
        for col in range(len(matrix[0]))
    ]
