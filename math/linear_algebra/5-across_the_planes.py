#!/usr/bin/env python3

"""
This script adds two matrices element wise.

"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
    mat1 (list of lists of ints/floats): The first matrix.
    mat2 (list of lists of ints/floats): The second matrix.

    Returns:
    list of lists of ints/floats: A new matrix containing
    the element-wise sum of mat1 and mat2.
    None: If mat1 and mat2 do not have the same shape.

    """
    # Check if the matrices have the same number of rows
    if len(mat1) != len(mat2):
        return None

    # Check if the matrices have the same number of columns in each row
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # Perform element-wise addition
    return [
        [elem1 + elem2 for elem1, elem2 in zip(row1, row2)]
        for row1, row2 in zip(mat1, mat2)
    ]
