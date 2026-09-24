import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """

    # (n_samples, n_features)
    centered_data = (X - np.mean(X, axis=0))
    cov = np.cov(centered_data, rowvar=False,ddof=0)
    
    values, vectors = np.linalg.eigh(cov)

    best_val = np.argsort(values)[::-1][:n_components]
    # (n_features, n_components)
    best_vectors = vectors[:, best_val]

    # (n_samples, n_features) @ (n_features, n_components) = (n_samples, n_components)
    proj = centered_data @ best_vectors

    reconstructed = (proj @ best_vectors.T) + np.mean(X, axis=0)

    mse = np.mean((X - reconstructed) ** 2)

    return mse
