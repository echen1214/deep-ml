import numpy as np

def softmax(array):
    # 1. Numerical Stability
    max_val = np.max(array, axis=1, keepdims=True)
    exp_array = np.exp(array - max_val)
    # 2. Row-wise Sum (axis=1) for Attention Weights
    sum_exp = np.sum(exp_array, axis=1, keepdims=True)
    return exp_array / sum_exp

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q, K, V)

def self_attention(Q, K, V):
    attention_output = softmax(Q @ K.T / np.sqrt(K.shape[0])) @ V
    return attention_output