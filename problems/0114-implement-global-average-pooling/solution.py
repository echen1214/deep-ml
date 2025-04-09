import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
    flat = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
    return(np.mean(flat, axis=0))
	# pass