import numpy as np
from typing import Tuple

def gini(y_subset):
    if y_subset.size == 0:
        return 0
    p = y_subset.mean()
    return 1 - (p**2 + (1-p)**2)

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    n, d = X.shape
    best_feature, best_threshold = -1, float('inf')
    best_gini = float('inf')

    for i in range(d):
        for threshold in np.unique(X[:, i]):
            l = y[X[:, i] <= threshold]
            r = y[X[:, i] > threshold]
            gini_l, gini_r = gini(l), gini(r)
            g_split = gini_l * (len(l)/n) + gini_r *(len(r)/n)
            if g_split < best_gini:
                best_gini = g_split
                best_feature = i 
                best_threshold = threshold
    return best_feature, best_threshold
