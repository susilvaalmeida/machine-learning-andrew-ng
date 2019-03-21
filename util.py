"""
Module with my implementations of utils methods used in programming exercises of Machine Learning course on Coursera.

Author: Suellen Silva de Almeida (susilvalmeida@gmail.com)
"""

import numpy as np

def add_intercept_term_to_X(X):
    """
    Add an additional first column to X to allow to treat theta_0 as simply another feature.

    Args:
        X (list): matrix of features of size (m, n) where m if the number of training examples and
                  n is the number of features
    Returns:
        (np.array): matrix of features of size (m, n+1) where m if the number of training examples and
                    n+1 is the number of features plus the intercept term column
    """
    X_new = np.ones(shape=(X.shape[0], X.shape[1]+1), dtype=float)
    X_new[:,1:] = X
    return X_new

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features.
    """
    degree = 6
    
    X1 = np.array(X1).reshape(-1,1)
    X2 = np.array(X2).reshape(-1,1)
    
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            p = (X1**(i-j)) * (X2**j)
            out = np.append(out, p, axis=1)
    return out