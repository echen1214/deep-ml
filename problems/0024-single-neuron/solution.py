import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
    def sigmoid(values):
        return np.divide(1, (1+np.exp(-values)))

    def forward(features, weights, bias):
        return np.dot(features, weights) + bias
    
    probabilities = sigmoid(forward(features, weights, bias))
    mse = np.mean(np.square(probabilities-np.asarray(labels)))
	return probabilities, mse