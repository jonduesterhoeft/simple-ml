import math
import pytest

from src.wizardml.classifiers.linear_models import linear_regression as l

# TEST PREDICT
def test_predict():
    alpha = 2.0
    beta = 0.5
    x_i = 10.0
    expected_result = 7.0
    result = l.predict(alpha, beta, x_i)
    assert pytest.approx(result) == expected_result
    
# TEST ERROR
def test_error():
    alpha = 2.0
    beta = 0.5
    x_i = 10.0
    y_i = 5.0
    expected_result = 2.0
    result = l.error(alpha, beta, x_i, y_i)
    assert pytest.approx(result) == expected_result

# TEST SUM_OF_SQUARES_ERROR
def test_sum_of_squares_error():
    alpha = 2.0
    beta = 0.5
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 4.0, 5.0, 6.0]
    expected_result = 7.5
    result = l.sum_of_squares_error(alpha, beta, x, y)
    assert pytest.approx(result, abs=1e-5) == expected_result
    
# TEST FIT_LEAST_SQUARES
def test_fit_least_squares():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 4.0, 5.0, 6.0]
    expected_result = (2.0, 1.0)
    result = l.fit_least_squares(x, y)
    assert pytest.approx(result[0]) == expected_result[0]
    assert pytest.approx(result[1]) == expected_result[1]

# TEST TOTAL_SUM_OF_SQUARES
def test_total_sum_of_squares():
    y = [1.0, 2.0, 3.0, 4.0]
    expected_result = 5.0
    result = l.total_sum_of_squares(y)
    assert pytest.approx(result) == expected_result

# TEST R_SQUARED
def test_r_squared():
    alpha = 2.0
    beta = 0.5
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 4.0, 5.0, 6.0]
    expected_result = -0.5
    result = l.r_squared(alpha, beta, x, y)
    assert pytest.approx(result) == expected_result
    
def test_r_squared_perfect_fit():
    alpha = 1.0
    beta = 0.2
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 4.0, 5.0, 6.0]
    expected_result = 1.0
    result = l.r_squared(alpha, beta, x, y)
    assert pytest.approx(result) == expected_result


if __name__ == '__main__':
    pass