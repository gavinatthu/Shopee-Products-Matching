"""Evaluate the prediction of model given results. 
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


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
        if np.isin(y_true[i], pred[:topN]).any():
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



def modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, *method):
    """Evaluate the model by calculatint F1-score, AUC or mAP corresponding to topN accuracy. 

    Evaluate the model by calculatint F1-score, AUC and mAP corresponding to topN accuracy. 
    Top-N predictions are given by the max-N dot products of test set features and training set features. 

    Args:
        testFeat: test set features predicted by model, # of test samples by # of features array
        testLabel: test set labels (ground truths), # of test samples array
        trainFeat: training set features predicted by model, # of training samples by # of features array
        trainLabel: training set labels (ground truths), # of training samples array
        method: {'F1', 'AUC', 'mAP'}, if use F1, topN should be provided. if not, topN == 1. Example: modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'F1', 2)

    Returns:
        eval: evaluation result corresponding to the supposed criteria
    """
    cor = np.dot(testFeat, trainFeat.T)
    y_true = testLabel
    if method[0] == 'F1':
        if len(method) == 1:
            topN = 1
        else:
            topN = method[1]
        top_index = np.argsort(-cor, axis=1)[:,:topN]
        y_pred = (np.asarray(trainLabel).T)[top_index]
        eval = f1_score(y_true, conv2Top1(y_true, y_pred, topN), average='micro')
    elif method[0] == 'AUC':
        y_true_multiLabel = np.zeros((len(testLabel),int(max(max(trainLabel), max(testLabel)))+1))
        for i, testSample in enumerate(y_true):
            y_true_multiLabel[i,testSample] = 1
        predict_proba = np.array([row/sum(row) for row in cor])
        y_pred = np.zeros((len(testLabel),int(max(max(trainLabel), max(testLabel)))+1))
        for i, col in enumerate(predict_proba.T):
            y_pred[:,trainLabel[i]] += col
        eval = roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro', labels=[i for i in range(int(max(max(trainLabel), max(testLabel)))+1)])
    elif method[0] == 'mAP':
        y_true_multiLabel = np.zeros((len(testLabel),int(max(max(trainLabel), max(testLabel)))+1))
        for i, testSample in enumerate(y_true):
            y_true_multiLabel[i,testSample] = 1
        predict_proba = np.array(cor)
        y_pred = np.zeros((len(testLabel),int(max(max(trainLabel), max(testLabel)))+1))
        for i, col in enumerate(predict_proba.T):
            y_pred[:,trainLabel[i]] += col
        eval = average_precision_score(y_true_multiLabel, y_pred, average='micro')
    return eval



# test case
if __name__ == '__main__':
    # y_true = np.asarray([1, 1, 2, 3])
    # y_pred = np.asarray([[1, 2],
    #                     [2, 1], 
    #                     [1, 3], 
    #                     [2, 3]])
    # print(conv2Top1(y_true, y_pred, topN=2))
    # print(calAccuracy(y_true, y_pred, topN=2))
    # print(calF1score(y_true, y_pred, topN=2))
    # print(conv2Top1(y_true, y_pred, topN=1))
    # print(calAccuracy(y_true, y_pred, topN=1))
    # print(calF1score(y_true, y_pred, topN=1))
    testFeat = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0.9, 0.1, 0]])
    # testLabel = np.asarray([0, 1, 2, 1])
    testLabel = [2, 3, 4, 9]
    trainFeat = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [1, 0.2, 0.2]])
    # trainLabel = np.asarray([0, 1, 2, 2, 1])
    trainLabel = [2, 3, 4, 5, 3]
    print(modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'mAP'))
