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
    expected_result = 2  # Partial derivative of a linear function is a constant
    result = g.partial_difference_quotient(linear_function, v, i, h)
    assert expected_result == pytest.approx(result, abs=1e-4) 

def test_partial_difference_quotient_quadratic():
    v = [1.0, -2.0, 3.0]
    i = 2
    h = 0.0001
    expected_result = -1  # Partial derivative of a quadratic function is a constant
    result = g.partial_difference_quotient(quadratic_function, v, i, h)
    assert expected_result == pytest.approx(result, abs=1e-4)

# TEST ESTIMATE_GRADIENT
def test_estimate_gradient_linear():
    v = [1.0, 2.0, 3.0]
    h = 0.001
    expected_result = [2.0, 2.0, 2.0]  # Gradient of a linear function is a constant vector
    result = g.estimate_gradient(linear_function, v, h)
    assert expected_result == pytest.approx(result, abs=1e-4)

def test_estimate_gradient_quadratic():
    v = [1.0, -2.0, 3.0]
    h = 0.0001
    expected_result = [2.0, 3.0, -1.0]  # Gradient of a quadratic function
    result = g.estimate_gradient(quadratic_function, v, h)
    assert expected_result == pytest.approx(result, abs=1e-4)

# TEST GRADIENT_STEP
def test_gradient_step_linear():
    v = [1.0, 2.0, 3.0]
    gradient = [2.0, 2.0, 2.0]
    step_size = 0.1
    expected_result = [1.2, 2.2, 3.2]
    result = g.gradient_step(v, gradient, step_size) 
    assert expected_result == pytest.approx(result, abs=1e-4) 
    
def test_gradient_step_quadratic():
    v = [1.0, -2.0, 3.0]
    gradient = [2.0, 3.0, -1.0]
    step_size = 0.01
    expected_result = [1.02, -1.97, 3.01]
    result = g.gradient_step(v, gradient, step_size)    
    assert expected_result == pytest.approx(result, abs=1e-4) 