import random
from typing import List
from ...math.linear_algebra.vector import Vector
from ...math.linear_algebra import vector as v
from ...math.stats.stats import corr, std, mean, subtract_mean
from ...math.gradient_descent import gradient_descent as g

# TODO
# Replace use linear algebra to solve regression instead of the gradient descent
# Add lasso regression

def predict(x: Vector, beta: Vector) -> float:
    """
    Predicts y values for a vector of x values using linear regression.
    
    y_i = beta_1*x_i1 + beta_2*x_i2 + ... + beta_k*x_ik + (alpha) + error_i
    beta = [beta_1, beta_2, ..., beta_k, (alpha)]
    x_i = [x_i1, xi_2, ..., x_ik, (1)]
    
    Parameters
    ----------
    beta : Vector
        Vector with beta values for the linear model.
    x : Vector
        The value of x for which we want to predict y.

    Returns
    -------
    float
        Returns the predicted value of y given x and beta parameters.
    """
    return v.dot(x, beta)

def error(x: Vector,  y: Vector, beta: float) -> float:
    """
    Calculates the error of the prediction against a known actual.
    
    Parameters
    ----------
    x : Vector
        The value of x for which we want to predict y.
    y : Vector
        The known value of y at x.
    beta : Vector
        Vector with beta values for the linear model.

    Returns
    -------
    float
        The error in prediction as y_predicted - y_actual.
    """
    return predict(x, beta) - y

def squared_error(x: Vector, y: Vector, beta: float) -> float:
    """
    Calculates the sum of squared errors from a simple linear regression.
    
    Parameters
    ----------
    x : Vector
        The value of x for which we want to predict y.
    y : Vector
        The known value of y at x.
    beta : Vector
        Vector with beta values for the linear model.
    
    Returns
    -------
    float
        The sum of squared error for the linear regression model.
    """
    return error(x, y, beta) ** 2

def squared_error_gradient(x: Vector, y: Vector, beta: float) -> float:
    """
    Calculates the sum of squared errors from a simple linear regression.
    
    Parameters
    ----------
    x : Vector
        The value of x for which we want to predict y.
    y : Vector
        The known value of y at x.
    beta : Vector
        Vector with parameter values for the linear model.
        
    Returns
    -------
    float
        The gradient vector of the squared errors.
    """
    error_val = error(x, y, beta)
    return [2 * error_val * x_i for x_i in x]

def fit_least_squares_gradient(x_vals: List[Vector],
                               y_vals: List[Vector],
                               learning_rate: float = 0.001,
                               num_steps: int = 1000,
                               batch_size: float | int = 1,
                               fit_intercept: bool = True) -> Vector:
    """
    Estimates the parameters for a linear regression using gradient descent.
    
    Parameters
    ----------
    x_vals : List[Vector]
        A list of vectors x_i for each point in the data set.
    y_vals : List[Vector]
        A list of vectors y_i for each point in the data set.
    learning_rate: float = 0.001
        The size of each gradient step.
    num_steps: int = 1000
        The number of gradient steps to make.
    batch_size: float | int = 1
        The number of minibatches for use in the gradient descent.
    fit_intercept: bool = True
        If true, appends a "1" to each vector in x_vals for the intercept.

    Returns
    -------
    Vector
        A vector of estimated parameters for the linear regression model.
    """
    assert len(x_vals) == len(y_vals), "X and Y vectors must be of equal length."
    # If we are fitting an intercept, add "1" to vals
    if fit_intercept:
        for val in x_vals:
            val.append(1.0)
    
    # Guess a random starting point
    beta_est = [random.random() for val in x_vals]
    
    # Perform a minibatch gradient descent for num_steps to estimate beta
    for _ in num_steps:
        batch_x = g.minibatch(x_vals, batch_size)
        batch_y = g.minibatch(y_vals, batch_size)
        while batch_x:
            gradient = v.vector_mean([squared_error_gradient(x, y, beta_est) for x, y in zip(batch_x, batch_y)])
            beta_est = g.gradient_step(beta_est, gradient, -learning_rate)
            
    return beta_est

def ridge_penalty(beta: Vector, alpha: float, fit_intercept: bool = True) -> float:
    """
    Adds a penalty term to the linear regression proportional to the sum of the squares
    of beta. If the intercept is being fitted, then the constant term is ignored.

    Parameters
    ----------
    beta : Vector
        Vector with parameter values for the linear model.
    alpha : float
        Hyperparameter determing how harsh the ridge penalty is.
    fit_intercept : bool, optional
        If true, ignore the last value (constant) of the beta vector.

    Returns
    -------
    float
        The value of the ridge penalty added to the error term.
    """
    if fit_intercept:
        beta = beta[:-1]  # Don't use the constant term
    return alpha * v.dot(beta, beta)

def ridge_squared_error(x: Vector, y:Vector, beta: Vector, alpha: float, fit_intercept: bool = True) -> float:
    """
    The regular squared error term plus the squared ridge penalty.

    Parameters
    ----------
    x : Vector
        The value of x for which we want to predict y.
    y : Vector
        The known value of y at x.
    beta : Vector
        Vector with parameter values for the linear model.
    alpha : float
        Hyperparameter determing how harsh the ridge penalty is.
    fit_intercept : bool, optional
        If true, ignore the last value (constant) of the beta vector.

    Returns
    -------
    float
        The regular squared error term plus the squared ridge penalty.
    """
    return squared_error(x, y, beta) + ridge_penalty(beta, alpha, fit_intercept)

def ridge_penalty_gradient(beta: Vector, alpha: float, fit_intercept: bool = True) -> Vector:
    """
    Calculates the gradient of the ridge penalty.

    Parameters
    ----------
    beta : Vector
        Vector with parameter values for the linear model.
    alpha : float
        Hyperparameter determing how harsh the ridge penalty is.
    fit_intercept : bool, optional
        If true, ignore the last value (constant) of the beta vector.

    Returns
    -------
    Vector
        The gradient of the ridge penalty.
    """
    if fit_intercept:
        beta = beta[:-1]  # Don't use the constant term
    return [0.] + [2 * alpha * beta_i for beta_i in beta]

def ridge_squared_error_gradient(x: Vector, y:Vector, beta: Vector, alpha: float, fit_intercept: bool = True) -> Vector:
    """
    Calculates the gradient of the squared errors and ridge penalty.

    Parameters
    ----------
    x : Vector
        The value of x for which we want to predict y.
    y : Vector
        The known value of y at x.
    beta : Vector
        Vector with parameter values for the linear model.
    alpha : float
        Hyperparameter determing how harsh the ridge penalty is.
    fit_intercept : bool, optional
        If true, ignore the last value (constant) of the beta vector.

    Returns
    -------
    Vector
        The gradient of the squared errors and ridge penalty.
    """
    return v.add(squared_error_gradient(x, y, beta) + ridge_penalty_gradient(beta, alpha, fit_intercept))

def fit_least_squares_ridge(x_vals: List[Vector],
                            y_vals: List[Vector],
                            learning_rate: float = 0.001,
                            num_steps: int = 1000,
                            batch_size: float | int = 1,
                            fit_intercept: bool = True) -> Vector:
    """
    Estimates the parameters for a linear regression using gradient descent.
    This version uses ridge regression which adds an error penalty proportional
    to the sum of the squares of beta_i.
    
    Parameters
    ----------
    x_vals : List[Vector]
        A list of vectors x_i for each point in the data set.
    y_vals : List[Vector]
        A list of vectors y_i for each point in the data set.
    learning_rate: float = 0.001
        The size of each gradient step.
    num_steps: int = 1000
        The number of gradient steps to make.
    batch_size: float | int = 1
        The number of minibatches for use in the gradient descent.
    fit_intercept: bool = True
        If true, appends a "1" to each vector in x_vals for the intercept.

    Returns
    -------
    Vector
        A vector of estimated parameters for the linear regression model.
    """
    assert len(x_vals) == len(y_vals), "X and Y vectors must be of equal length."
    # If we are fitting an intercept, add "1" to vals
    if fit_intercept:
        for val in x_vals:
            val.append(1.0)
    
    # Guess a random starting point
    beta_est = [random.random() for val in x_vals]
    
    # Perform a minibatch gradient descent for num_steps to estimate beta
    for _ in num_steps:
        batch_x = g.minibatch(x_vals, batch_size)
        batch_y = g.minibatch(y_vals, batch_size)
        while batch_x:
            gradient = v.vector_mean([ridge_squared_error_gradient(x, y, beta_est, fit_intercept) for x, y in zip(batch_x, batch_y)])
            beta_est = g.gradient_step(beta_est, gradient, -learning_rate)
            
    return beta_est

def total_sum_of_squares(y: Vector) -> float:
    """
    The total sum of variations of y_i's from their mean.
    
    Parameters
    ----------
    y : Vector
        A vector of y_i's.

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
    return 1.0 - (squared_error(x, y, beta) / total_sum_of_squares(y))


if __name__ == '__main__':
    pass