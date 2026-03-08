import numpy as np
import math

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1, Z2 = map(np.array, (Z1, Z2))
    S = np.matmul(Z1, Z2.T) / temperature
    loss = 0
    S = S - np.max(S, axis=1, keepdims=True)
    n, _  = S.shape
    for i in range(n):
        den = 0
        for j in range(n):
            den += math.exp(S[i, j])
        num = math.exp(S[i, i])
        loss += math.log(num / den)
    return -1/n * loss
                
            
            