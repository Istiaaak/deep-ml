import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    x_t = np.asarray(X_train)
    y_t = np.asarray(y_train)
    X_t = x_t ** np.arange(degree + 1)

    x_v = np.asarray(X_val)
    y_v = np.asarray(y_val)
    X_v = x_v ** np.arange(degree +1)

    res = {"train_errors": [], "val_errors": [], "diagnosis": ""}
    for train_size in train_sizes:
        X_temp = X_t[:train_size]
        y_temp = y_t[:train_size]


        w = np.linalg.pinv(X_temp) @ y_temp
        n_samples, _ = X_temp.shape
        n_val, _ = X_v.shape
        train_loss = (1/n_samples) * np.sum((X_temp @ w - y_temp)**2)
        val_loss = (1/n_val) * np.sum((X_v @ w - y_v)**2)
        res["train_errors"].append(train_loss)
        res["val_errors"].append(val_loss)

    if res['train_errors'][-1] > bias_threshold:
        res['diagnosis'] = "high_bias"
    elif res['val_errors'][-1] - res['train_errors'][-1] > variance_threshold:
        res['diagnosis'] = "high_variance"
    else:
        res['diagnosis'] = 'good_fit'
    return res