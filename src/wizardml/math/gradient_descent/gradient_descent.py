import random
from typing import Callable, TypeVar, List, Iterator
from ..linear_algebra import vector as vector
from ..linear_algebra.vector import Vector

def partial_difference_quotient(f: Callable[[Vector], float], 
                                v: Vector,
                                i: int, 
                                h: float) -> float:
    """
    Calculates the ith partial difference quotient for a function
    f(v) at v.

    Parameters
    ----------
    f : Callable[[Vector, float]] 
        The function for which we want the partial difference quotient
    v : Vector
        The point at which the difference quotient is calculated
    i : int
        The element of v for which we are calculating the quotient
    h : float
        The difference over which we calculate the quotient

    Returns
    -------
    float
        The ith partial difference quotient for a function f at v.
    """
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float], 
                                v: Vector,
                                h: float = 0.0001) -> Vector:
    """
    Estimates the gradient of function f(v) at v.

    Parameters
    ----------
    f : Callable[[Vector, float]] 
        The function for which we want the partial difference quotient
    v : Vector
        The point at which the difference quotient is calculated
    h : float (default to 0.0001)
        The difference over which we calculate the quotient

    Returns
    -------
    Vector
        The estimated gradient of a function f at v.
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Moves a distance of step_size in the direction of the gradient from v.

    Parameters
    ----------
    v : Vector
        The vector starting point
    gradient : Vector
        The gradient for a function f at v
    step_size : float 
        The distance to move from v.

    Returns
    -------
    Vector
        A new vector moved step_size along the gradient from v.
    """
    assert len(v) == len(gradient)
    step = vector.scalar_multiply(gradient, step_size)
    return vector.add(v, step)

T = TypeVar('T')  ## Generic Function

def minibatch(dataset: List[T], 
              batch_size: int | float, 
              shuffle: bool = True) -> Iterator:
    """
    Breaks a dataset into 'minibatchs' for use in gradient descent.

    Parameters
    ----------
    dataset: List[T]
        A dataset as a list of objects of type T.
    batch_size : Vector
        The size of the minibatches. A float between 0 and 1 is treated
        as a percentage and multiplied by the size of the dataset to
        determine the batch_size. Otherwise it specifies the number of
        samples in each minibatch.
    shuffle : float 
        Determines whether or not to randomize the order of batches.

    Returns
    -------
    Vector
        A new vector moved step_size along the gradient from v.
    """
    # If batch_size in (0,1) treat as a percentage of the dataset
    data_size = len(dataset)
    if 0 < batch_size < 1:
        batch_size = batch_size * data_size
    batch_size = int(batch_size)
    assert batch_size > 0, "batch_size must be greater than 0"
    
    
    batch_starts = [start for start in range(0, data_size, batch_size)]
    
    if shuffle:
        random.shuffle(batch_starts)
        
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]