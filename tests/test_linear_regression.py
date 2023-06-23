import math
import pytest

from src.wizardml.classifiers.linear_models import linear_regression as l

# TEST PREDICT
def test_predict():
    x = [1.0, 2.0, 3.0]
    beta = [0.5, 0.25, 0.75]
    expected_result = 3.25
    result = l.predict(x, beta)
    assert pytest.approx(result) == expected_result
    
# TEST ERROR
def test_error():
    x = [1.0, 2.0, 3.0]
    y = 5.0
    beta = [0.5, 0.25, 0.75]
    expected_result = -1.75
    result = l.error(x, y, beta)
    assert pytest.approx(result) == expected_result

# TEST SQUARED_ERROR
def test_squared_error():
    x = [1.0, 2.0, 3.0]
    y = 5.0
    beta = [0.5, 0.25, 0.75]
    expected_result = [-3.5, -7, -10.5]
    result = l.squared_error(x, y, beta)
    assert pytest.approx(result) == expected_result
    
# TEST SQUARED_ERROR_GRADIENT
def test_squared_error_gradient():
    x = [1.0, 2.0, 3.0]
    y = 5.0
    beta = [0.5, 0.25, 0.75]
    expected_result = 3.0625
    result = l.squared_error(x, y, beta)
    assert pytest.approx(result) == expected_result

# TEST FIT_LEAST_SQUARES
# def test_fit_least_squares():
    # x = [1.0, 2.0, 3.0, 4.0]
    # y = [3.0, 4.0, 5.0, 6.0]
    # expected_result = (2.0, 1.0)
    # result = l.fit_least_squares(x, y)
    # assert pytest.approx(result[0]) == expected_result[0]
    # assert pytest.approx(result[1]) == expected_result[1]

# TEST TOTAL_SUM_OF_SQUARES
# def test_total_sum_of_squares():
    # y = [1.0, 2.0, 3.0, 4.0]
    # expected_result = 5.0
    # result = l.total_sum_of_squares(y)
    # assert pytest.approx(result) == expected_result

# TEST R_SQUARED
# def test_r_squared():
#     alpha = 2.0
#     beta = 0.5
#     x = [1.0, 2.0, 3.0, 4.0]
#     y = [3.0, 4.0, 5.0, 6.0]
#     expected_result = -0.5
#     result = l.r_squared(alpha, beta, x, y)
#     assert pytest.approx(result) == expected_result
    
# def test_r_squared_perfect_fit():
#     alpha = 2.0
#     beta = 1.0
#     x = [1.0, 2.0, 3.0, 4.0]
#     y = [3.0, 4.0, 5.0, 6.0]
#     expected_result = 1.0
#     result = l.r_squared(alpha, beta, x, y)
#     assert pytest.approx(result) == expected_result


if __name__ == '__main__':
    pass