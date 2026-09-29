def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	# Your code here
    updated_params = []
    
    for param_array, grad_array, apply_wd in zip(parameters, gradients, apply_to_all):
        updated_array = []
        for param, grad in zip(param_array, grad_array):
            if apply_wd:

                updated_param = param - lr * grad - lr * weight_decay * param
            else:
                updated_param = param - lr * grad
            updated_array.append(updated_param)
        updated_params.append(updated_array)
    
    return updated_params