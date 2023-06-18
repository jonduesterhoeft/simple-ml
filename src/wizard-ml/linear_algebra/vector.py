import math
from typing import List

# Define Vector type
Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """
    Add two vectors of equal length.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].
    w : Vector
        A Vector of type List[float].

    Returns
    -------
    Vector
        The componentwise vector addition of v and w.
    """
    # Check that vectors are of equal length
    assert len(v) == len(w), 'Vectors must be of equal size'
    return [vi + wi for vi, wi in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtract two vectors of equal length.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].
    w : Vector
        A Vector of type List[float].

    Returns
    -------
    Vector
        The componentwise vector subtraction of v and w.
    """
    # Check that vectors are of equal length
    assert len(v) == len(w), 'Vectors must be of equal size'
    return [vi - wi for vi, wi in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Componentwise sum of a list of vectors.

    Parameters
    ----------
    vectors : List[Vector]
        A List of Vectors of type List[Vector]

    Returns
    -------
    Vector
        A componentwise sum of all vectors in the list vectors.
    """
    # Check that vectors exists
    assert vectors, 'Must pass a list of vectors.'
    # Check that all vectors are of equal length
    vector_length = len(vectors[0])  # Use length of first vector
    size_text = 'Vectors must all be of equal size.'
    assert all(len(v) == vector_length for v in vectors), size_text
    return [sum(v[i] for v in vectors) for i in range(vector_length)]


def scalar_multiply(v: Vector, c: float = 1.0) -> Vector:
    """
    Multiply a Vector v, by a scalar c.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].
    c : float, optional
        A scalar value, by default 1.0

    Returns
    -------
    Vector
        A Vector of type List[float].
    """
    return [c * vi for vi in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Componentwise means of a list of vectors.

    Parameters
    ----------
    vectors : List[Vector]
        A List of Vectors of type List[Vector]

    Returns
    -------
    Vector
        A vector of componentwise means of all vectors.
    """
    # Check that vectors exists
    assert vectors, 'Must pass a list of vectors.'
    count_vectors = len(vectors)
    return scalar_multiply(vector_sum(vectors), (1 / count_vectors))


def dot(v: Vector, w: Vector) -> float:
    """
    Calculate the dot product of two vectors v and w.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].
    w : Vector
        A Vector of type List[float].

    Returns
    -------
    float
        The scalar dot product of v and w.
    """
    # Check that vectors are of equal length
    assert len(v) == len(w), 'Vectors must be of equal size'
    return sum(vi * wi for vi, wi in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    """
    Calculates of the sum of squares of a vector v's components.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].

    Returns
    -------
    float
        The sum of squares of vector v's components
    """
    return dot(v, v)


def magnitude(v: Vector) -> float:
    """
    Calculates the magnitude (or length) of a vector v.

    Parameters
    ----------
    v : Vector
        A Vector of type List[float].

    Returns
    -------
    float
        The magnitude (or length) of a vector v.
    """
    return math.sqrt(sum_of_squares(v))


def distance(v: Vector, w: Vector) -> float:
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
    return magnitude(subtract(v, w))
