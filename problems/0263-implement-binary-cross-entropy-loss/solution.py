# Cell 1
import numpy as np
def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    # Your code here
    out = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    out = -(np.multiply(y_true, np.log(y_pred)) + np.multiply((1-y_true), np.log(1-y_pred)))
    # print(out)
    mean = np.sum(out) / (len(out) + epsilon)
    return mean
    # pass