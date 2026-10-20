import torch
from torch.utils.data import TensorDataset, DataLoader
import math
def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    num_batches = math.ceil(X.shape[0]/4)
    X_batch, y_batch = next(iter(loader))
    return num_batches, tuple(X_batch.shape)
