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
	
def multi_head_attention(Q, K, V, n_heads):
    D_model = Q.shape[1]
    stride = D_model/n_heads
    out = []
    for i in range(n_heads):
        ind1 = int(i*stride)
        ind2 = int((i+1)*stride)
        # print(ind1, ind2)
        # split matrices along the column
        o = self_attention(Q[:,ind1:ind2], K[:,ind1:ind2], V[:,ind1:ind2])
        out.append(o)
    return np.concatenate(out, axis=1)
