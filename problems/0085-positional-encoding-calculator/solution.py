import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
    if position == 0:
        return -1
    if d_model <= 0:
        return -1

    pos_row = np.arange(position)[:,np.newaxis]
    scale = np.power(10000, 2*(np.arange(d_model)[np.newaxis, :]//2)/d_model)
    x = pos_row/scale
    x[:,::2] = np.sin(x[:,::2])
    x[:,1::2] = np.cos(x[:,1::2])
        # pos_encoding = np.float16(pos_encoding)
    return np.float16(x)