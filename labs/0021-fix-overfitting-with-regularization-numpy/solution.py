import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    
    var = np.var(X_train, axis=0)
    X_train_new = X_train[:, var > 1e-3]
    X_val_new = X_val[:, var > 1e-3]
    alpha = 0.1
    I = np.eye(X_train_new.shape[1])
    beta = np.linalg.inv(X_train_new.T @ X_train_new + alpha * I) @ X_train_new.T @ y_train


    def predict(X):
        X_new = X[:, var > 1e-3]
        return X_new @ beta
    
    return predict