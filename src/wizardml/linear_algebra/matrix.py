import math
from typing import List, Tuple, Callable

import vector as v

# Define Matrix type
Matrix = List[List[float]]
Vector = v.Vector


def display_matrix(matrix: Matrix) -> None:
    """
    Display the matrix in a standard row x column view.

    Parameters
    ----------
    matrix : Matrix
        A matrix of the type List[List[float]].

    Returns
    -------
    None
    """
    for row in matrix:
        print(row, end='\n')
    return None


def shape(matrix: Matrix) -> Tuple[int, int]:
    """
    Returns the number of rows and columns in a matrix.

    Parameters
    ----------
    matrix : Matrix
        A matrix of type List[List[float]].

    Returns
    -------
    Tuple[int, int]
        The shape of the matrix in the form of (rows, columns).
    """
    if not matrix:
        return (0, 0)
    rows = len(matrix)
    if len(matrix[0]) == 0:
        return (0, 0)
    columns = len(matrix[0]) if matrix else 0
    return (rows, columns)


def get_row(matrix: Matrix, row_i: int = None) -> Vector:
    """
    Returns a specific row of a matrix.
    If no index is provided, function will generate rows from 0 to n.

    Parameters
    ----------
    matrix : Matrix
        A matrix of type List[List[float]].
    row_i : int
        The row index to return.
        If none, then return a generator for all rows.

    Returns
    -------
    Vector
        A vector of type List[float].

    Yields
    -------
    Iterator[Vector]
        [description]
    """
    # Check that matrix exists
    assert matrix, 'Must pass a matrix.'
    rows = len(matrix)
    if row_i is not None:  # Return row as vector
        assert row_i <= rows - 1, 'Row index out of bounds.'
        return matrix[row_i]
    else:  # Return generator if no row index specified
        return iter(matrix[i] for i in range(rows))


def get_column(matrix: Matrix, col_j: int = None) -> Vector:
    """
    Returns a specific column of a matrix.
    If no index is provided, function will generate columns from 0 to n.

    Parameters
    ----------
    matrix : Matrix
        A matrix of type List[List[float]].
    row_i : int
        The row index to return.
        If none, then return a generator for all rows.

    Returns
    -------
    Vector
        A vector of type List[float].

    Yields
    -------
    Iterator[Vector]
        [description]
    """
    # Check that matrix exists
    assert matrix, 'Must pass a matrix.'
    rows = len(matrix)
    cols = len(matrix[0])
    if cols == 0:
        return []
    if col_j is not None:  # Return column as vector
        assert col_j <= cols - 1, 'Row index out of bounds.'
        return [matrix[i][col_j] for i in range(rows)]
    else:  # Return generator if no column index specified
        return iter([matrix[i][j] for i in range(rows)] for j in range(cols))
    

def build_matrix(rows: int, columns: int, function: Callable) -> Matrix:
    """
    Builds a matrix of the specified size.
    Pass a function to construct the element based on their indicies.

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    columns : int
        Number of columns in the matrix.
    function : Callable
        A function to calculate the individual elements of the matrix.

    Returns
    -------
    Matrix
        Returns a matrix of the specified shape using the passed function
        to calculate element values.
    """
    return [[function(i, j) for i in range(columns)] for j in range(rows)]


def identity_matrix(n: int = 1) -> Matrix:
    """
    Returns an n x n identity matrix.

    Parameters
    ----------
    n : int
        Size of the n x n identity matrix.

    Returns
    -------
    Matrix
        A matrix with values of 1 on the diagonal, otherwise zero.
    """
    if n < 1:
        return [[]]
    size = range(n)
    return [[1 if i == j else 0 for i in size] for j in size]


if __name__ == '__main__':
    pass
