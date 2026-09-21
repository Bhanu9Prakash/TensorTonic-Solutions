import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    d = lambda a, b, c: a*b +c 
    d_a = (d(a+h, b, c) - d(a, b, c))/h
    d_b = (d(a, b+h, c) - d(a, b, c))/h
    d_c = (d(a, b, c+h) - d(a, b, c))/h

    return d(a, b, c), d_a, d_b, d_c
