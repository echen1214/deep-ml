import numpy as np

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = sum(y_true[np.where(y_pred==1)] == 1)
    fp = sum(y_true[np.where(y_pred==1)] == 0)
    tn = sum(y_true[np.where(y_pred==0)] == 0)
    fn = sum(y_true[np.where(y_pred==0)] == 1)

    if (tp + fp) == 0 or (tp + fn) == 0: 
        return round(0.0,3)
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1,3)
    # return round(0,3)