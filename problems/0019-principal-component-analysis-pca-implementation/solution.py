import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	# Your code here
    data = (data - np.mean(data,axis=0))/ np.std(data,axis=0)
    cov = np.cov(data.T)
    eigv, principal_components = np.linalg.eig(cov)

    sort = principal_components[:, np.argsort(eigv)[::-1]]
	return np.round(sort[:,[np.arange(0,k)]], 4)