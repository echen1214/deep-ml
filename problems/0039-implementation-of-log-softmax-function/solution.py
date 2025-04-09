import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here 
	return np.log(np.divide(np.exp(scores), np.sum(np.exp(scores))))