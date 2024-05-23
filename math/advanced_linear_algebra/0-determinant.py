#!/usr/bin/env python3


def determinant(matrix):
    """Calculates the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The determinant of the matrix, or 1 for a 0x0 matrix.

    Raises:
        TypeError: If the input is not a list of lists.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    rows = len(matrix)
    if rows != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Base cases: 0x0 and 1x1 matrices
    if rows == 0:
        return 1  # Special case for 0x0 matrix
    elif rows == 1:
        return matrix[0][0]

    # Recursive case: Use Laplace expansion
    det = 0
    for col in range(rows):
        # Get the minor matrix excluding the first row and current column
        minor = [[matrix[i][j] for j in range(rows) if j != col] for i in range(1, rows)]
        sign = (-1) ** col  # Alternate signs for Laplace expansion
        det += sign * matrix[0][col] * determinant(minor)

    return det
