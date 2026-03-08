import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X)
    
    if X.ndim == 1:
        X = X.reshape(1, -1)

    m, n = X.shape

    if m == 1:
        return None
    X_norm = X - np.mean(X, axis = 0, keepdims = True)
    cov = np.zeros((n, n))
    deviated = np.sum(X_norm ** 2, axis = 0) / (m - 1)
    
    for i in range(n):
        for j in range(n):
            cov[i, j] = np.sum(X_norm[:, i] * X_norm[:, j]) / (m - 1)
            
    return cov
        