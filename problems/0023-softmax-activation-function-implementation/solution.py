import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
	# Your code here
    probabilities = np.divide(np.exp(scores),np.sum(np.exp(scores)))
	return probabilities