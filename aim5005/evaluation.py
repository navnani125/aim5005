import numpy as np
from typing import List

def predict(intercept: float, beta: float, x_i: float) -> float:
    """
    Predicts y value based on intercept, beta (slope), and x value.
    """
    return intercept + beta * x_i

def error(intercept: float, beta: float, x_i: float, y_i: float) -> float:
    """
    Find the difference between the predicted value and the actual value, where actual is y_i.
    """
    return predict(intercept, beta, x_i) - y_i

def sum_of_square_error(intercept: float, beta: float, x: List[float], y_actual: List[float]) -> float:
    """
    Square the errors and sum them.
    """
    return np.sum([error(intercept, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y_actual)])

def total_sum_of_squares(y_actual: List[float]) -> float:
    """
    Calculate the total sum of squares (SST).
    """
    meanval = np.mean(y_actual)
    return np.sum([(y - meanval) ** 2 for y in y_actual])

def rsquared(intercept: float, coef: float, x: List[float], y_actual: List[float]) -> float:
    """
    R^2 = 1 - (SSE/SST)
    """
    sse = sum_of_square_error(intercept, coef, x, y_actual)
    sst = total_sum_of_squares(y_actual)
    return 1 - (sse / sst)
