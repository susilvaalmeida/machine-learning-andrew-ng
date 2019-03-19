"""
Module with my linear regression implementation using gradient descent and normal equation based on programming
exercise 1 of Machine Learning course on Coursera.

Author: Suellen Silva de Almeida (susilvalmeida@gmail.com)
"""

import numpy as np

def compute_cost_one_variable(X, y, theta):
    """
    Computes the cost (squared error function) for linear regresion with one variable.
    That is, computes the cost of using theta as the parameter for linear regression to fit
    the data points in X and y.
    Formulas:
        h(x) = theta_0 + theta_1 * x 
        J(theta_0, theta_1) = 1/2m * sum(h(x) - y)^2)

    Args:
        X (np.array): vector of feature of size (m, 1) where m is the number of training examples
        y (np.array): vector with target variable of size (m, 1) where m is the number of training examples
        theta (np.array): vector of parameters of linear regression of size (n+1, 1) where n is the number of features
    Return:
        float: The cost of using theta as the parameter for linear regression to fit
               the data points in X and y.
    """
    m = y.shape[0]
    h = X.dot(theta)
    J = np.multiply((1/(2*m)), np.sum((h - y)**2))
    return J

def compute_cost(X, y, theta):
    """
    Computes the cost (squared error function) for linear regresion with multiple variables.
    That is, computes the cost of using theta as the parameter for linear regression to fit
    the data points in X and y. 
    Formula: J(theta) = 1/2m * (X*theta - y)' * (X*theta - y)

    Args:
        X (np.array): matrix of features of size (m, n+1) where m if the number of training examples and 
                      n is the number of features (+1 is the intercept term)
        y (np.array): vector with target variable of size (m, 1) where m is the number of training examples
        theta (np.array): vector of parameters of linear regression of size (n+1, 1) where n is the number of features
    Return:
        float: The cost of using theta as the parameter for linear regression to fit
               the data points in X and y.
    """
    m = y.shape[0]
    h = X.dot(theta)
    J = np.multiply((1/(2*m)), (h-y).T.dot(h-y))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta (linear regression parameters) by taking num_iters gradient steps with
    learning rate alpha.
    Formulas:
        h(x) = theta_0 + theta_1*x(1) + theta_2*x(2) + ... + theta_n*x(n) 
        theta(j) = theta(j) - alpha*(1/m)*sum(h(x)-y)*x(j), for j=0,...n
    
    Args:
        X (np.array): matrix of features of size (m, n+1) where m if the number of training examples and 
                      n is the number of features (+1 is the intercept term)
        y (np.array): vector with target variable of size (m, 1) where m is the number of training examples
        theta (np.array): vector of parameters of linear regression of size (n+1, 1) where n is the number of features
        alpha (float): the learning rate
        num_iters (int): number of iterations of gradient descent
    Returns:
        np.array: computed theta (vector of parameters of linear regression)
        np.array: vector with the cost of each iteration
    """
    m = y.shape[0]
    J_history = np.zeros(shape=(num_iters, 1))
    
    for i in range(0, num_iters):
        h = X.dot(theta)
        diff_hy = h - y
        
        delta = np.multiply((1/m), diff_hy.T.dot(X))
        theta = theta - np.multiply(alpha, delta.T)
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

def normal_eqn(X, y):
    """
    Computes the closed-form solution to linear regression using normal equations.
    Formula:
        theta = (X'*X)^(-1) * X' * y

    Args:
        X (np.array): matrix of features of size (m, n+1) where m if the number of training examples and 
                      n is the number of features (+1 is the intercept term)
        y (np.array): vector with target variable of size (m, 1) where m is the number of training examples
    Returns:
        np.array: computed theta (vector of parameters of linear regression)
    """
    inv = np.linalg.pinv(X.T.dot(X))
    theta = inv.dot(X.T).dot(y)
    return theta

def fit(X, y, fit_method='gradient_descent', learning_rate=0.1, num_iterations=50, normalize=False):
    """
    Fit the linear regression model, that is, compute best value of parameteres theta.
    The values of theta can be found using gradient descent or normal equation.
    If it uses gradient descent, performs gradient descent to learn theta (linear regression parameters) by taking 
    num_iterations gradient steps with learning rate learning_rate.

    Args:
        X (list): matrix of features of size (m, n) where m if the number of training examples and
                  n is the number of features
        y (list): vector with target variable of size m where m is the number of training examples
        fit_method (str): name of the method that will be use to find linear regression paramenters, can be
                          gradient_descent or normal_equation
        learning_rate (float): the learning rate of gradient descent
        num_iterations (int): number of iterations of gradient descent
        normalize (bool): True to normalize the features
    Returns:
        dict: Dictionary with matrix X, theta values, list of cost for each iteration of gradient descent, 
              list of mean and standard deviation of each column of X if normalize is True
    """

    X = np.array(X, dtype=float)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if normalize:
        X, mean, std = feature_normalize(X)

    X = add_intercept_term_to_X(X)
    y = np.array(y, dtype=float).reshape(-1,1)
    theta = np.zeros(shape=(X.shape[1],1))
    J_history = None

    if fit_method == 'gradient_descent':
        theta, J_history = gradient_descent(X, y, theta, learning_rate, num_iterations)
    elif fit_method == 'normal_equation':
        theta = normal_eqn(X, y)

    fitted_model = {
        'X': X,
        'theta': theta,
        'J_history': J_history,
    }

    if normalize:
        fitted_model['mean'] = mean
        fitted_model['std'] = std

    return fitted_model

def predict(X, fitted_model):
    """
    Uses the fitted_model dict generated in fit method to predict the target for input X.
    If the fitted_model normalized the features, use the same mean and std to normalize the input features.

    Args:
        X (list): vector of features of size (1, n) n is the number of features
        fitted_model (dict): Dictionary with matrix X (used in fit), theta values, list of cost for each iteration of
                             gradient descent, and list of mean and standard deviation of each column of X if normalize
                             is True
    Returns:
        float: predicted value for features X
    """
    X = np.array(X, dtype=float)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if 'mean' in fitted_model and 'std' in fitted_model:
        X, _, _ = feature_normalize(X, fitted_model['mean'], fitted_model['std'])

    X = add_intercept_term_to_X(X)

    return X.dot(fitted_model['theta'])

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
    for i in range(1, X.shape[1]+1):
        X_new[:, i] = X[:, i-1]
    return X_new

def feature_normalize(X, mean=[], std=[]):
    """
    Normalizes the features in X by subtracting the mean value of each feature from the dataset and dividing the
    features values by their respective standard deviations. Most data points will lie within +-2 standard deviations
    of the mean. If the mean and std were computed in a previous step (like in fit method), use the computed values.

    Args:
        X (list): matrix of features
        mean (list): mean of each feature if it was already computed
        std (list): std of each feature if it was already computed
    Returns:
        (np.array): matrix of features X with normalized values
        (np.array): vector with mean of each feature
        (np.array): vector with std of each feature
    """
    X_norm = X

    if len(mean) == 0 and len(std) == 0:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)

    for i in range(0, X.shape[1]):
        X_norm[:, i] = (X_norm[:, i] - mean[i])/std[i]
    return X_norm, mean, std
