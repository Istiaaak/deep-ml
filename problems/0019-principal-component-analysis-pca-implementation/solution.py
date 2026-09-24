import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    normalized_data = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    cov = np.cov(normalized_data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_eigenvalues = np.argsort(eigenvalues)[-k:][::-1]

    sorted_vectors = eigenvectors[:, sorted_eigenvalues]

    for i in range(sorted_vectors.shape[1]):
        col = sorted_vectors[:, i]
        valid_indices = np.where(np.abs(col) > 1e-10)[0]
        if len(valid_indices) > 0:
            if col[valid_indices[0]] < 0:
                sorted_vectors[:, i] *= -1
    
    return np.round(sorted_vectors, 4)
    





