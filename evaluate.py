"""Evaluate the prediction of model given results. 

TODO: implement AUC, mAP"""

import numpy as np
from sklearn.metrics import f1_score

def calAccuracy(y_true, y_pred, topN=1):
    """Calculate Accuracy. 

    Calculate Accuracy corresponding to topN accuracy. 

    Args:
        y_true: Labels
        y_pred: Model predictions of size len(y_true)*n sorted by decreasing grades, n>=topN
        topN: N in top-N

    Returns:
        Acc: Accuracy
    """
    Acc = 0
    yn = len(y_true)
    for i, pred in enumerate(y_pred):
        if np.isin(y_true[i], pred[:topN]):
            Acc += 1
    return Acc/yn

def conv2Top1(y_true, y_pred, topN=1):
    """Convert top-N predictions to top-1 predictions. 

    Convert top-N predictions to top-1 predictions. If top-N hits, replace the N predictions by the correct label; 
    If not hit, replace by the first label (with highest grade)

    Args:
        y_true: Labels
        y_pred: Model predictions of size len(y_true)*n sorted by decreasing grades, n>=topN
        topN: N in top-N

    Returns:
        y_pred_top1: Top-1 predictions
    """
    y_pred_top1 = np.array(y_pred[:,0]) # copy 1st column of y_pred
    for i, pred in enumerate(y_pred):
        if np.isin(y_true[i], pred[:topN]):
            y_pred_top1[i] = y_true[i]
    return y_pred_top1
    
def calF1score(y_true, y_pred, topN=1, average='micro'):
    """Calculate F1-score corresponding to topN accuracy. 

    Calculate F1-score corresponding to topN accuracy by converting top-N predictions to top-1 predictions. 
    Calculate metrics globally by counting the total true positives, false negatives and false positives, 
    which means all classes are considered and the denominator of F1 is the total number of prediction. 
    Under such assumption, F1-score == accuracy. 

    Args:
        y_true: Labels
        y_pred: Model predictions of size len(y_true)*n sorted by decreasing grades, n>=topN
        topN: N in top-N
        average : {'micro', 'macro', 'samples','weighted', 'binary'} or None

    Returns:
        f1Score: F1-score
    """
    return f1_score(y_true, conv2Top1(y_true, y_pred, topN), average=average)

# test case
if __name__ == '__main__':
    y_true = np.asarray([1, 1, 2, 3])
    y_pred = np.asarray([[1, 2],
                        [2, 1], 
                        [1, 3], 
                        [2, 3]])
    print(conv2Top1(y_true, y_pred, topN=2))
    print(calAccuracy(y_true, y_pred, topN=2))
    print(calF1score(y_true, y_pred, topN=2))
    print(conv2Top1(y_true, y_pred, topN=1))
    print(calAccuracy(y_true, y_pred, topN=1))
    print(calF1score(y_true, y_pred, topN=1))
