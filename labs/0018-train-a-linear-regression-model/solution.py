import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    lr = 0.1
    n_iter=500
    n = X.shape[0]
    delta_w = (2/n) * X.T @ (X @ W + b - y)
    delta_b = (2/n) * np.sum(X @ W + b - y)

    for i in range(n_iter):
        W-= lr * delta_w
        b-= lr * delta_b

    return W, b
