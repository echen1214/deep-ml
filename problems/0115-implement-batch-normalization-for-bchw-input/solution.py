import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	# Your code here
    return gamma * np.divide((X - np.mean(X, axis=(0,2,3),keepdims=True)), (np.sqrt(np.var(X,axis=(0,2,3),keepdims=True)+epsilon))) + beta
	# pass