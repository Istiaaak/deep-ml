import numpy as np

def sigmoid(z):
	return 1/(1+np.exp(-z))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	features = np.asarray(features, dtype=float)
	labels = np.asarray(labels, dtype=float)
	updated_weights = np.asarray(initial_weights, dtype=float).copy()
	updated_bias = float(initial_bias)
	n = features.shape[0]
	mse_values = []
	for _ in range(epochs):
		z = features @ updated_weights.T + updated_bias
		pred = sigmoid(z)
		mse = np.mean((pred - labels)**2)
		mse_values.append(mse)

		grad_w = (2 / n) * features.T @ ((pred - labels) * pred * (1 - pred))
		grad_b = (2 / n) * np.sum((pred - labels)*(1 - pred)*pred)

		updated_weights-= learning_rate * grad_w
		updated_bias -= learning_rate * grad_b

	return (
		np.round(updated_weights, 4),
		round(updated_bias, 4),
		[round(mse, 4) for mse in mse_values]
	)