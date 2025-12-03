import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(array):
    # 1. Numerical Stability
    max_val = np.max(array, axis=1, keepdims=True)
    exp_array = np.exp(array - max_val)
    # 2. Row-wise Sum (axis=1) for Attention Weights
    sum_exp = np.sum(exp_array, axis=1, keepdims=True)
    return exp_array / sum_exp

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
    score = (Q @ K.T / np.sqrt(K.shape[0])) + mask
    attention_output = softmax(score) @ V
    return attention_output
