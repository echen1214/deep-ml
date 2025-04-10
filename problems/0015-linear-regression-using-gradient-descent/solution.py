import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	def linear_func(X, theta):
        return np.dot(X, theta)
        # return theta[0]*X[0]+theta[1]*X[1]
    m, n = X.shape
	theta = np.zeros((n, 1))
    
    for i in range(0,iterations):
        theta = theta - alpha * np.mean((np.dot(X, theta) - y.reshape(3,1)) * X, axis=0).reshape(2,1)

	return np.round(theta,4)