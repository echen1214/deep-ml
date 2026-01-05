import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    correct_class = predicted_probs[true_labels==1]
    cel = -(np.sum(np.log(correct_class)) / (len(correct_class) + epsilon))
    if cel:
        return cel
    else:
        return 0.0
    pass#