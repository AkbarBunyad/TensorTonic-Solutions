import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    a, p, n = map(np.array, (anchor, positive, negative))
    d_pos = np.sum((a - p)**2, axis = -1)
    d_neg = np.sum((a - n)**2, axis = -1)
    loss = np.maximum(0, d_pos - d_neg + margin)
    return float(np.mean(loss))
    