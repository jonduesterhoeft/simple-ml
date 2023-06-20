import pytest
import math

from src.wizardml.math.linear_algebra import matrix as m


# TEST SHAPE
def test_shape_null():
    a = None
    assert m.shape(a) == (0, 0)
    
def test_shape_empty():
    a = [[]]
    assert m.shape(a) == (0, 0)
    
def test_shape_row():
    a = [[1, 2, 3]]
    assert m.shape(a) == (1, 3)
    
def test_shape_column():
    a = [[1], [2], [3]]
    assert m.shape(a) == (3, 1)
    
def test_shape_matrix():
    a = [[1, 2, 3], [1, 1, 1], [1, 1, 1]]
    assert m.shape(a) == (3, 3)


# TEST GET_ROW
def test_get_row_null():
    a = None
    with pytest.raises(AssertionError, match=r'.*pass a matrix.*'):
        m.get_row(a)
    
def test_get_row_empty():
    a = [[]]
    assert m.get_row(a, 0) == []
    
def test_get_row_oob():
    a = [[]]
    with pytest.raises(AssertionError, match=r'.*out of bounds.*'):
        m.get_row(a, 1)
    
def test_get_row_row():
    a = [[1, 2, 3]]
    assert m.get_row(a, 0) == [1, 2, 3]
    
def test_get_row_column():
    a = [[1], [2], [3]]
    assert m.get_row(a, 1) == [2]
    
def test_get_row_matrix():
    a = [[1, 2, 3], [1, 1, 1], [1, 1, 1]]
    assert m.get_row(a, 2) == [1, 1, 1]
    
def test_get_row_generator():
    a = [[1, 2, 3], [1, 1, 1], [1, 1, 1]]
    assert list(m.get_row(a)) == a
    
    
# TEST GET_COLUMN
def test_get_column_null():
    a = None
    with pytest.raises(AssertionError, match=r'.*pass a matrix.*'):
        m.get_column(a)
    
def test_get_column_empty():
    a = [[]]
    assert m.get_column(a, 0) == []
    
def test_get_column_oob():
    a = [[1], [2], [3]]
    with pytest.raises(AssertionError, match=r'.*out of bounds.*'):
        m.get_column(a, 1)
    
def test_get_column_row():
    a = [[1, 2, 3]]
    assert m.get_column(a, 0) == [1]
    
def test_get_column_column():
    a = [[1], [2], [3]]
    assert m.get_column(a, 0) == [1, 2, 3]
    
def test_get_column_matrix():
    a = [[1, 2, 3], [1, 1, 1], [1, 1, 1]]
    assert m.get_column(a, 2) == [3, 1, 1]
    
def test_get_column_generator():
    a = [[1, 2, 3], [1, 1, 1]]
    test = list(m.get_column(a))
    assert test[2] == [3, 1]


# TEST BUILD MATRIX
def test_build_matrix_zero():
    def zero(i, j):
        return 0
    assert m.build_matrix(1, 1, zero) == [[0]]

def test_build_matrix_zeroes():
    def zero(i, j):
        return 0
    assert m.build_matrix(2, 3, zero) == [[0, 0, 0], [0, 0, 0]]
    
def test_build_matrix_function():
    def add_function(i, j):
        return i + j
    assert m.build_matrix(2, 3, add_function) == (
        [[0, 1, 2], [1, 2, 3]])
    

# TEST IDENTITY MATRIX
def test_identify_matrix_null():
    assert m.identity_matrix(0) == [[]]

def test_identify_matrix_1():
    assert m.identity_matrix(1) == [[1]]

def test_identify_matrix_2():
    assert m.identity_matrix(2) == [[1, 0], [0, 1]]