#!/usr/bin/env python3

"""
This script concatenates two matrices along a specific axis.

"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Parameters:
    mat1 (list of lists of ints/floats): The first matrix.
    mat2 (list of lists of ints/floats): The second matrix.
    axis (int): The axis along which to concatenate
    (0 for rows, 1 for columns).

    Returns:
    list of lists of ints/floats: A new matrix that is
    the concatenation of mat1 and mat2 along the specified axis.
    None: If the two matrices cannot be concatenated.

    """
    if axis == 0:
        # Concatenate along rows
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Concatenate along columns
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        # Invalid axis
        return None
