import numpy as np
import math

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    output_height = int(math.ceil((input_height + padding - kernel_height)/stride) + 1)
    output_width = int(math.ceil((input_width + padding - kernel_width)/stride) + 1)
    output_matrix = np.zeros((output_height, output_width))
    
	# Your code here
    padded_matrix = np.pad(input_matrix, ((padding,padding), (padding,padding)), mode='constant')
    for i in range(output_width):
        for j in range(output_height):
            row_inp_start = (i*stride)
            row_inp_end = (i*stride)+kernel_width
            height_inp_start = (j*stride)
            height_inp_end = (j*stride)+kernel_width
            # print(i,j, input_matrix[inp_start:inp_end, inp_start:inp_end])
            # print(inp_start, inp_end)
            output_matrix[i][j] = np.sum(kernel * 