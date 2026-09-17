
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	n=y_pred.size
	rmse = np.sum((y_true - y_pred)**2)
	rmse_res= np.sqrt((1/n)*rmse)
	return round(rmse_res,3)
