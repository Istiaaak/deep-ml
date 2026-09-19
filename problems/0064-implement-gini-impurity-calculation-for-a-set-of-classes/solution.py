
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	n = len(y)
	res={}
	for item in y:	
		res[item] = res.get(item, 0) + 1
	
	val = 1
	for k, v in res.items():
		val-= (v/n)**2
	return round(val,3)