import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Your code here
    l_recons = np.mean(np.sum(np.square(x-x_reconstructed), axis=1))

    kl_div = (-1/2) * np.mean(
        np.sum(1 + log_var - np.square(mu) - np.exp(log_var), axis=1)
    )

    loss = l_recons + kl_div

    return float(loss), float(l_recons), float(kl_div)