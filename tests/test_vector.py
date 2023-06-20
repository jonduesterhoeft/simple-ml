import math
import pytest

from src.wizardml.math.linear_algebra import vector as v

# VECTOR ADD TESTS
def test_add_unequal():
    a = [1, 2]
    b = [1]
    with pytest.raises(AssertionError, match=r'.*equal size.*'):
        v.add(a, b)

def test_add_positive():
    a = [1, 2]
    b = [3, 4]
    assert v.add(a, b) == [4, 6]

def test_add_negative():
    a = [-1, 2]
    b = [3, -4]
    assert v.add(a, b) == [2, -2]

def test_add_zero():
    a = [1, 2]
    b = [0, 0]
    assert v.add(a, b) == [1, 2]
    
def test_add_self():
    a = [1, 2]
    assert v.add(a, a) == [2, 4]


# VECTOR SUBTRACT TESTS
def test_subtract_unequal():
    a = [1, 2]
    b = [1]
    with pytest.raises(AssertionError, match=r'.*equal size.*'):
        v.subtract(a, b)

def test_subtract_positive():
    a = [1, 2]
    b = [3, 4]
    assert v.subtract(a, b) == [-2, -2]

def test_subtract_negative():
    a = [-1, 2]
    b = [3, -4]
    assert v.subtract(a, b) == [-4, 6]

def test_subtract_zero():
    a = [1, 2]
    b = [0, 0]
    assert v.subtract(a, b) == [1, 2]
    
def test_subtract_self():
    a = [1, 2]
    assert v.subtract(a, a) == [0, 0]
    
    
# VECTOR SUM TESTS
def test_vector_sum_empty():
    vectors = []
    with pytest.raises(AssertionError, match=r'.*list of vectors*'):
        v.vector_sum(vectors)
    
def test_vector_sum_unequal():
    a = [1, 2]
    b = [1]
    vectors = [a, b]
    with pytest.raises(AssertionError, match=r'.*equal size.*'):
        v.vector_sum(vectors)

def test_vector_sum_positive():
    a = [1, 2]
    b = [3, 4]
    vectors = [a, b]
    assert v.vector_sum(vectors) == v.add(a, b)

def test_vector_sum_negative():
    a = [-1, 2]
    b = [3, -4]
    vectors = [a, b]
    assert v.vector_sum(vectors) == v.add(a, b)

def test_vector_sum_zero():
    a = [1, 2]
    b = [0, 0]
    vectors = [a, b]
    assert v.vector_sum(vectors) == v.add(a, b)
    
def test_vector_sum_self():
    a = [1, 2]
    vectors = [a, a]
    assert v.vector_sum(vectors) == v.add(a, a)
    
def test_vector_sum_many():
    a = [1, 1, 1]
    b = [2, 3, 4]
    c = [0, 0, 0]
    d = [-1, -1, -1]
    vectors = [a, b, c, d]
    assert v.vector_sum(vectors) == [2, 3, 4]


# VECTOR MULTIPLY TESTS
def test_multiply_default():
    a = [1, 1]
    assert v.scalar_multiply(a) == a

def test_multiply_positive():
    c = 2
    a = [1, 2]
    assert v.scalar_multiply(a, c) == [2, 4]

def test_multiply_negative():
    c = -2
    a = [1, 2]
    assert v.scalar_multiply(a, c) == [-2, -4]

def test_multiply_zero():
    c = 0
    a = [1, 2]
    assert v.scalar_multiply(a, c) == [0, 0]
    
    
# VECTOR MEAN TESTS
def test_vector_mean_empty():
    vectors = []
    with pytest.raises(AssertionError, match=r'.*list of vectors*'):
        v.vector_mean(vectors)
    
def test_vector_mean_unequal():
    a = [1, 2]
    b = [1]
    vectors = [a, b]
    with pytest.raises(AssertionError, match=r'.*equal size.*'):
        v.vector_mean(vectors)

def test_vector_mean_positive():
    a = [1, 2]
    b = [3, 4]
    vectors = [a, b]
    assert v.vector_mean(vectors) == [2, 3]

def test_vector_mean_negative():
    a = [-1, 2]
    b = [3, -4]
    vectors = [a, b]
    assert v.vector_mean(vectors) == [1, -1]

def test_vector_mean_zero():
    a = [1, 2]
    b = [0, 0]
    vectors = [a, b]
    assert v.vector_mean(vectors) == [0.5, 1]
    
def test_vector_mean_self():
    a = [1, 2]
    vectors = [a, a]
    assert v.vector_mean(vectors) == a
    
def test_vector_mean_many():
    a = [1, 1, 1]
    b = [2, 3, 4]
    c = [0, 0, 0]
    d = [-1, -1, -1]
    vectors = [a, b, c, d]
    assert v.vector_mean(vectors) == [0.5, 0.75, 1]
    

# DOT PRODUCT TESTS
def test_dot_unequal():
    a = [1, 2]
    b = [1]
    with pytest.raises(AssertionError, match=r'.*equal size.*'):
        v.dot(a, b)

def test_dot_positive():
    a = [1, 2]
    b = [3, 4]
    assert v.dot(a, b) == 11

def test_dot_negative():
    a = [-1, 2]
    b = [3, -4]
    assert v.dot(a, b) == -11

def test_dot_zero():
    a = [1, 2]
    b = [0, 0]
    assert v.dot(a, b) == 0
    
def test_dot_self():
    a = [1, 2]
    assert v.dot(a, a) == 5
    
    
# TEST SUM OF SQUARES
def test_sum_of_squares_zero():
    a = [0, 0]
    assert v.sum_of_squares(a) == 0
    
def test_sum_of_squares_one():
    a = [1, 1]
    assert v.sum_of_squares(a) == 2
    
def test_sum_of_squares_positive():
    a = [1, 2, 3]
    assert v.sum_of_squares(a) == 14
    
def test_sum_of_squares_negative():
    a = [-1, -2, -3]
    assert v.sum_of_squares(a) == 14
    
    
# TEST MAGNITUDE
def test_magnitude_zero():
    a = [0, 0]
    assert v.magnitude(a) == 0
    
def test_magnitude_one():
    a = [1, 1]
    assert v.magnitude(a) == math.sqrt(2)
    
def test_magnitude_positive():
    a = [1, 2, 3]
    assert v.magnitude(a) == math.sqrt(14)
    
def test_magnitude_negative():
    a = [-1, -2, -3]
    assert v.magnitude(a) == math.sqrt(14)
    
def test_magnitude_float():
    a = [1, 0.5, 0]
    assert v.magnitude(a) == math.sqrt(1.25)
    
    
# TEST DISTANCE
def test_distance_zero():
    a = [0, 0]
    b = [0, 0]
    assert v.distance(a, b) == 0
    
def test_distance_zero_one():
    a = [1, 1]
    b = [0, 0]
    assert v.distance(a, b) == math.sqrt(2)
    
def test_distance_positive():
    a = [1, 2, 3]
    b = [3, 2, 1]
    assert v.distance(a, b) == math.sqrt(8)
    
def test_distance_negative():
    a = [-1, -2, -3]
    b = [3, 2, 1]
    assert v.distance(a, b) == math.sqrt(48)
    
def test_distance_float():
    a = [1, 0.5, 0]
    b = [0, 0, 0]
    assert v.distance(a, b) == math.sqrt(1.25)