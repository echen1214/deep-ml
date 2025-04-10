import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.asarray(X)
	theta = np.round(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y), 4)

	return theta