import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    rng = np.random.default_rng(seed)
    n, d = X_train.shape

    stumps = []

    for _ in range(n_estimators):
        indices = rng.choice(n, size=n, replace=True)

        X_boot = X_train[indices]
        y_boot = y_train[indices]

        best_stump = {
            "feature": None,
            "threshold": None,
            "polarity": None,
            "error": 1.0
            }

        for j in range(d):
            features_values = X_boot[:, j]
            unique_values = np.unique(features_values)

            thresholds = (unique_values[1:] + unique_values[:-1])/2

            for t in thresholds:
                pred = (features_values > t).astype(int)
                error = np.mean(pred != y_boot)

                pred_inverse = 1 - pred
                error_inverse = np.mean(pred_inverse != y_boot)

                if error < best_stump['error']:
                    best_stump["error"] = error
                    best_stump["polarity"] = 1
                    best_stump["threshold"] = t
                    best_stump["feature"] = j

                if error_inverse < best_stump['error']:
                    best_stump["error"] = error_inverse
                    best_stump["polarity"] = -1
                    best_stump["threshold"] = t
                    best_stump["feature"] = j
            
            stumps.append(best_stump)
        

        all_preds = []
        for stump in stumps:
            feature = stump['feature']
            threshold = stump['threshold']
            
            values = X_test[:,feature]
            pred = (values > threshold).astype(int)

            if stump['polarity'] == -1:
                pred = 1 - pred
            
            all_preds.append(pred)
        
        predictions = np.mean(np.asarray(all_preds), axis=0)

        
        return (predictions >= 0.5).astype(int)
            
                