import numpy as np
from scipy.spatial.distance import cdist

def l2_normalise(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def pdist_dot(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    return -(a @ b.T)

def pdist_euclidean(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    return cdist(a, b, metric='euclidean').astype(np.float32)