from typing import Tuple

from ...math.linear_algebra.vector import Vector
from ...math.stats.stats import corr, std, mean, subtract_mean

def predict(alpha: float, beta: float, x_i: float) -> float:
    """
    Predicts using a simple linear regression model.
    y_i = beta*x_i + alpha + error_i
    
    Parameters
    ----------
    alpha : float
        alpha term for simple linear regression.
    beta : float
        beta term for simple linear regression.
    x_i : float
        The value of x for which we want to predict y.

    Returns
    -------
    float
        Returns the predicted value of y at x.
    """
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    Calculates the error of the prediction against a known actual.
    
    Parameters
    ----------
    alpha : float
        alpha term for simple linear regression.
    beta : float
        beta term for simple linear regression.
    x_i : float
        The value of x for which we want to predict y.
    y_i : float
        The known value of y at x.

    Returns
    -------
    float
        The error in prediction as y_predicted - y_actual
    """
    return predict(alpha, beta, x_i) - y_i

def sum_of_squares_error(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    Calculates the sum of squared errors from a simple linear regression.
    
    Parameters
    ----------
    alpha : float
        alpha term for simple linear regression.
    beta : float
        beta term for simple linear regression.
    x : Vector
        The values of x for which we want to predict values.
    y : Vector
        The actual values y for the given x.

    Returns
    -------
    float
        The sum of squared error for a simple linear regression
    """
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def fit_least_squares(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Calculates the alpha, beta terms for simple linear regression
    that minimizes the sum of squared errors
    
    Parameters
    ----------
    x : Vector
        The values of x for which we want to predict values.
    y : Vector
        The actual values y for the given x.

    Returns
    -------
    alpha, beta : float, float
        The alpha, beta terms that minimize the sum of squared errors
    """
    beta = corr(x, y) * std(y) / std(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y: Vector) -> float:
    """
    The total sum of variations of y_i's from their mean.
    
    Parameters
    ----------
    y : Vector
        fA vector of y_i's.

    Returns
    -------
    float
        The sum of squared values of y_i's with their mean subtracted.
    """
    return sum(v ** 2 for v in subtract_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    The fraction of variation in y explained by the model.
    
    Parameters
    ----------
    alpha : float
        alpha term for simple linear regression.
    beta : float
        beta term for simple linear regression.
    x : Vector
        The values of x for which we want to predict values.
    y : Vector
        The actual values y for the given x.

    Returns
    -------
    float
        The value calculated for R^2.
    """
    return 1.0 - (sum_of_squares_error(alpha, beta, x, y) / total_sum_of_squares(y))


if __name__ == '__main__':
    pass