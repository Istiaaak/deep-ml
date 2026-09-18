import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	tp = (y_true == 1) & (y_pred == 1)
	tn = (y_true == 0) & (y_pred == 0)
	num = (np.sum(tp) + np.sum(tn))
	return num/len(y_pred)