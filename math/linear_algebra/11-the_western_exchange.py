#!/usr/bin/env python3

"""
This script transposes matrix

"""


def np_transpose(matrix):
    """
    Transpose a matrix.

    Args:
        matrix (list): The input matrix.

    Returns:
        list: The transposed matrix.
    """
    if not matrix:
        return []

    return [list(row) for row in zip(*matrix)]
