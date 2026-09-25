import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	X = np.c_[np.ones((X.shape[0], 1)), X]
	c = max(y) + 1 
	n, m = X.shape
	beta = np.zeros((c, m))
	losses = []
	for _ in range(iterations):
		scores = X @ beta.T
		probs = softmax(scores)
		loss = -np.sum(np.log(probs[np.arange(n), y]))
		one_hot = np.zeros((n, c))
		one_hot[np.arange(n), y] = 1

		grad_beta = (probs - one_hot).T @ X
		beta-= learning_rate*grad_beta
		losses.append(loss)

	return (beta, losses)