#!/usr/bin/env python3

"""
This is the script that two arrays element-wise.

"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Parameters:
    arr1 (list of ints/floats): The first array.
    arr2 (list of ints/floats): The second array.

    Returns:
    list: A new list containing the element-wise sum of arr1 and arr2.
    None: If arr1 and arr2 are not the same shape.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
