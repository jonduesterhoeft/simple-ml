import math

def uniform_pdf(x: float) -> float:
    """
    The standard uniform probability density function.

    Parameters
    ----------
    x : float
        A value in the range [0, 1].

    Returns
    -------
    float
        1 for x in [0, 1].
    """
    return 1 if 0 <= x <= 1 else 0

def uniform_cdf(x: float) -> float:
    """
    The standard uniform cumulative density function.

    Parameters
    ----------
    x : float
        A value in the range [0, 1].

    Returns
    -------
    float
        The probability that a uniform random variable <= x.
    """
    if x < 0:
        return 0
    elif x <= 1:
        return x
    else:
        return 1


def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    """
    The normal probability density function.

    Parameters
    ----------
    x : float
        
    mu : float
        Mean for the distrubiton
        
    sigma : 
        Standard deviation for the distrubiton, must be >= 0

    Returns
    -------
    float
        The probability that a normal random variable = x.
    """
    if x <= 0:
        return None
    a = math.sqrt(2 * math.pi) * sigma
    b = (( x - mu) ** 2) / (2 * sigma ** 2)
    return math.exp(-b) / a


def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    """
    The normal cumulative probability density function.

    Parameters
    ----------
    x : float
        
    mu : float
        Mean for the distrubiton
        
    sigma : 
        Standard deviation for the distrubiton, must be >= 0

    Returns
    -------
    float
        The probability that a normal random variable <= x.
    """
    if x <= 0:
        return None
    return (1 + math.erf((x - mu) / (math.sqrt(2) * sigma))) / 2


# TODO - Add inverse normal cdf