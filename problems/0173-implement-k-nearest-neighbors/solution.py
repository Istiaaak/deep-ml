import numpy as np

def dist_euc(p, q):
    return np.sqrt(np.sum((p-q)**2,axis=1))

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    pts = np.asarray(points)
    query_pt = np.asarray(query_point)

    dist = dist_euc(query_pt, pts)
    sort = np.argsort(dist, kind='stable')
    return list(map(tuple, pts[sort[:k]]))







    





