import numpy as np
def precision(y_true, y_pred):
	# Your code here
	tp = (y_true == 1) & (y_pred == 1)
	fp = (y_true == 0) & (y_pred == 1)
	if np.sum(fp)+np.sum(tp) == 0:
		return 0
	return np.sum(tp)/(np.sum(tp)+np.sum(fp))