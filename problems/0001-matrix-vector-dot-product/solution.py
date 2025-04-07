def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a) != len(b):
		return -1
	out = []

	for a1 in a:
		sum = 0
		for id, i in enumerate(a1):
			sum += i*b[id]
		out.append(sum) 
	return out
	# pass