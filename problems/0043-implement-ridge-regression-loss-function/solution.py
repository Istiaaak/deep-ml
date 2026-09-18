import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	n, d = X.shape
	mse = (1/n)* np.sum((X @ w - y_true)**2)
	ridge = alpha * w @ w
	return mse+ridge
