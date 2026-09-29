import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0
    for _ in range(max_iter):

        grad_w = (1/n) * X.T @ (X @ weights + bias - y) + alpha1*np.sign(weights) + alpha2 * 2 * weights
        grad_b = (1/n) * np.sum(X @ weights + bias - y)

        weights-=learning_rate*grad_w
        bias-=learning_rate*grad_b
        if np.linalg.norm(grad_w, ord=1) < tol:
            break
    return weights, bias
