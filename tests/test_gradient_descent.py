import pytest

from src.wizardml.math.gradient_descent import gradient_descent as g

# DEFINE TEST FUNCTIONS
# Define a linear function: f(v) = 2v + 1
def linear_function(v):
    return sum(2 * v_j for v_j in v) + 1

# Define a quadratic function: f(v) = v_0^2 + 3v_1 - v_2
def quadratic_function(v):
    return v[0]**2 + 3*v[1] - v[2]


# TEST PARTIAL_DIFFERENCE_QUOTIENT
def test_partial_difference_quotient_linear():
    v = [1.0, 2.0, 3.0]
    i = 1
    h = 0.001
    expected_result = 2.0 
    result = g.partial_difference_quotient(linear_function, v, i, h)
    assert pytest.approx(result) == expected_result

def test_partial_difference_quotient_quadratic():
    v = [1.0, -2.0, 3.0]
    i = 2
    h = 0.0001
    expected_result = -1.0 
    result = g.partial_difference_quotient(quadratic_function, v, i, h)
    assert pytest.approx(result) == expected_result

# TEST ESTIMATE_GRADIENT
def test_estimate_gradient_linear():
    v = [1.0, 2.0, 3.0]
    h = 0.001
    expected_result = [2.0, 2.0, 2.0]  
    result = g.estimate_gradient(linear_function, v, h)
    assert pytest.approx(result) == expected_result

def test_estimate_gradient_quadratic():
    v = [1.0, -2.0, 3.0]
    h = 0.0001
    expected_result = [2.0, 3.0, -1.0] 
    result = g.estimate_gradient(quadratic_function, v, h)
    assert pytest.approx(result, abs=2*h) == expected_result

# TEST GRADIENT_STEP
def test_gradient_step_linear():
    v = [1.0, 2.0, 3.0]
    gradient = [2.0, 2.0, 2.0]
    step_size = 0.1
    expected_result = [1.2, 2.2, 3.2]
    result = g.gradient_step(v, gradient, step_size) 
    assert pytest.approx(result) == expected_result
    
def test_gradient_step_quadratic():
    v = [1.0, -2.0, 3.0]
    gradient = [2.0, 3.0, -1.0]
    step_size = 0.01
    expected_result = [1.02, -1.97, 2.99]
    result = g.gradient_step(v, gradient, step_size)    
    assert pytest.approx(result) == expected_result
    
# TEST MINIBATCH
def test_minibatch_fixed_batch_size():
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 3
    expected_result = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    result = list(g.minibatch(dataset, batch_size, shuffle=False))
    assert result == expected_result

def test_minibatch_percentage_batch_size():
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 0.4
    expected_result = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
    result = list(g.minibatch(dataset, batch_size, shuffle=False))
    assert result == expected_result

def test_minibatch_shuffle():
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 2
    unexpected_result = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    result = list(g.minibatch(dataset, batch_size, shuffle=True))
    assert result != unexpected_result


if __name__ == '__main__':
    pass
