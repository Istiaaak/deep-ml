import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    normalized_data = (X - np.mean(X, axis=0))
    cov = np.cov(normalized_data, rowvar=False)

    values, vectors = np.linalg.eigh(cov)
    sorted_values = values[::-1]

    return sorted_values/np.sum(values)