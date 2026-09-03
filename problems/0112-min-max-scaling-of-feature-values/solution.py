def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    mini = min(x)
    maxi = max(x)
    new_x = [(item - mini)/(maxi - mini) for item in x]
    return new_x