import numpy as np

def cosine_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def knn_retrieve(Xn, idx, k=5):
    sims = Xn @ Xn[idx]
    nn = np.argsort(-sims)[:k]
    return nn, sims[nn]
