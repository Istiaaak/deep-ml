import numpy as np
def sig(x):
	return 1/(1+np.exp(-x))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	n, d = 	X.shape
	weigts = np.zeros(d)
	bias = 0
	losses = []
	for _ in range(iterations):
		
		z = X @ weigts + bias

		loss = - (y.T @ np.log(sig(z)) + (1-y) @ np.log(1-sig(z)))
		losses.append(np.round(loss, 4))
		# gradient descent
		weigts-= learning_rate * X.T @ (sig(z) - y) 
		bias-= learning_rate * np.sum(sig(z) - y)
		
	params = [round(float(bias), 4)] + np.round(weigts, 4).tolist()
	return params, losses

