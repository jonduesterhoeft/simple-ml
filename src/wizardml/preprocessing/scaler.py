from typing import Tuple, List

from ..math.linear_algebra.vector import Vector
from ..math.linear_algebra import vector as v
from ..math.stats import stats as stat


def scale(data: List[Vector]) -> Tuple(Vector, Vector):
    """
    Returns the mean and standard deviation of each position.

    Parameters
    ----------
    data : List[Vector]
        The dataset being scaled.
    
    Returns
    -------
    float
        The mean and standard deviation for each position of the dataset.
    """
    size = len(data[0])
    
    mean = v.vector_mean(data)
    stdev = [stat.std([vector[i] for vector in data]) for i in size]
    
    return mean, stdev


def rescale(data: List[Vector]) -> List[Vector]:
    """
    Standard scaler, rescales the dataset to zero mean and unit variance.

    Parameters
    ----------
    data : List[Vector]
        The dataset being scaled.

    Returns
    -------
    List[Vector]
        The rescaled dataset with zero mean and unit variance.
    """
    size = len(data[0])
    mean, stdev = scale(data)
    
    rescaled = [v[:] for v in data]  # Copy
    
    for v in rescaled:
        for i in range(size):
            if stdev[i] > 0:
                v[i] = (v[i] - mean[i]) / stdev[i]
                
    return rescaled