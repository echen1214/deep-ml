import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code here
	def twod_det(twod_mat):
		return (twod_mat[0][0] * twod_mat[1][1]) - (twod_mat[0][1] * twod_mat[1][0])

	matrix = np.array(matrix)
	if len(matrix)==2:
		det = twod_det(matrix)
	elif len(matrix)==3:
		det = 0
		for i in range(3):
			index = list(range(3))
			index.pop(i)
			if i % 2:
			    neg = -1
			else:
			    neg = 1
			det_sum = neg * matrix[0][i] * twod_det(matrix[[1,2]][:, index])
			# print(matrix[[1,2]][:, index], (-i%2),  twod_det(matrix[[1,2]][:, index]),  index, det_sum)
			det += det_sum

	trace_list = [matrix[i][i] for i in range(len(matrix))]

	trace = 0
	for t in trace_list:
		trace += t
		
	return det, trace