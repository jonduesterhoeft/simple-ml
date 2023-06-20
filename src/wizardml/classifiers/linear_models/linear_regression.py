from typing import Tuple

from ...math.linear_algebra.vector import Vector
from ...math.stats.stats import corr, std, mean, subtract_mean

def predict(alpha: float, beta: float, x_i: float) -> float:
    """
    Calculates the distance between two vectors v and w.
    
    Parameters
    ----------
    v : Vector
        A Vector of type List[float].
    w : Vector
        A Vector of type List[float].

    Returns
    -------
    float
        Returns the distance between two vectors v and w.
    """
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

def sum_of_squares_error(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def fit_least_squares(x: Vector, y: Vector) -> Tuple[float, float]:
    beta = corr(x, y) * std(y) / std(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in subtract_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_squares_error(alpha, beta, x, y) / total_sum_of_squares(y))

if __name__ == '__main__':
    pass