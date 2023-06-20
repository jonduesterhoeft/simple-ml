from typing import List
from collections import Counter
import math

from ..linear_algebra.vector import dot, sum_of_squares

def mean(x: List[float]) -> float:
    """
    Calculate the mean of a list of numbers.

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    float
        The mean of the list of values.
    """
    if len(x) == 0:  # Return None for empty lists
        return None
    return sum(x) / len(x)


def median(x: List[float]) -> float:
    """
    Calculate the median value of a list of numbers.

    Parameters
    ----------
    x : List[float]
        A list of numbers

    Returns
    -------
    float
        The median value of the list.
        For even numbered lists, the mean of the two middle elements is 
        returned.
    """
    
    def _median_odd(x):
        """Calculate median for odd-numbered lists."""
        return sorted(x)[len(x) // 2]
    
    def _median_even(x):
        """Calculate median for even-numbered lists."""
        x_sorted = sorted(x)
        midpoint_high = len(x) // 2
        return (x_sorted[midpoint_high - 1] + x_sorted[midpoint_high]) / 2
    
    if len(x) == 0:  # Return None for empty lists
        return None
    even = len(x) % 2 == 0  # Check if list has an even number of elements
    return _median_even(x) if even else _median_odd(x)


def quantile(x: List[float], p: float) -> float:
    """
    Returns the value from a list x such that p percent of values are below it.

    Parameters
    ----------
    x : List[float]
        A list of values.
    p : float
        The percent of values that are below the return value.

    Returns
    -------
    float
        The value in list x such that p percent of values in x are below.
    """
    if len(x) == 0:
        return None
    quantile_index = int(len(x) * p)
    return sorted(x)[quantile_index]


def mode(x: List[float]) -> List[float]:
    """
    Returns the most frequent value. If there are multiple elements with
    the same maximum fequency, return them all.

    Parameters
    ----------
    x : List[float]
        A list of numbers.

    Returns
    -------
    List[float]
        A list of the most frequent element(s) in x.
    """
    if len(x) == 0:
        return None
    counts = Counter(x)
    max_count = max(counts.values())
    return [xi for xi, count in counts.items() if count == max_count]


def range(x: List[float]) -> float:
    """
    Returns the range of a list of values.

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    float
        The distance between the maximum and minimum values in the list.
    """
    if len(x) == 0:
        return None
    return max(x) - min(x)


def subtract_mean(x: List[float]) -> List[float]:
    """
    Subtracts the mean value each element in a list of values.

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    List[float]
        The deviation of each element of x from the mean value of x.
    """
    if len(x) == 0:
        return None
    x_mean = mean(x)
    return [xi - x_mean for xi in x]


def variance(x: List[float]) -> float:
    """
    Returns the variance of a list of values.

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    float
        The variance of x.
    """
    if len(x) == 0:
        return None
    if len(x) == 1:
        return 0
    n = len(x)
    deviations = subtract_mean(x)
    return sum_of_squares(deviations) / (n - 1)


def std(x: List[float]) -> float:
    """
    Returns the standard deviation of a list of values.

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    float
        The standard deviation of x.
    """
    if len(x) == 0:
        return None
    return math.sqrt(variance(x))


def iqr(x: List[float]) -> float:
    """
    Return the Interquartile Range (IQR).

    Parameters
    ----------
    x : List[float]
        A list of values.

    Returns
    -------
    float
        The difference between the 75th and 25th percentiles.
    """
    if len(x) == 0:
        return None
    return quantile(x, p=0.75) - quantile(x, p=0.25)


def cov(x: List[float], y: List[float]) -> float:
    """
    Returns the sample covariance of x and y.

    Parameters
    ----------
    x : List[float]
        A list of values.
    y : List[float]
        A list of values.

    Returns
    -------
    float
        The sample covariance.
    """
    assert len(x) == len(y), 'Vectors must be of equal size.'
    if len(x) == 0:
        return None
    elif len(x) == 1:
        return 0
    return dot(subtract_mean(x), subtract_mean(y)) / (len(x) - 1)


def corr(x: List[float], y: List[float]) -> float:
    """
    Return the correlation coefficient of x and y.

    Parameters
    ----------
    x : List[float]
        A list of values.
    y : List[float]
        A list of values.

    Returns
    -------
    float
        The correlation coefficient of x and y.
    """
    if len(x) == 0 or len(y) == 0:
        return None
    std_x = std(x)
    std_y = std(y)
    if std_x <= 0 and std_y <= 0:
        return 0
    return cov(x, y) / (std_x * std_y)