import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_confusion_matrix(predictions:np.ndarray, y:np.ndarray)->np.ndarray:
    y_actual = pd.Series(y, name='Actual')
    y_predicted = pd.Series(predictions, name="Predicted")
    return pd.crosstab(y_actual, y_predicted).to_numpy()

def get_precision(confusion_matrix:np.ndarray, labels:list[str])->dict:
    precisions = {}
    for i in range(len(labels)):
        precisions[labels[i]] = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
    return precisions

def get_recall(confusion_matrix:np.ndarray, labels:list[str])->dict:
    recalls = {}
    for i in range(len(labels)):
        recalls[labels[i]] = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
    return recalls

def get_f1_score(precesions:dict, recalls:dict, labels:list[str])->dict:
    f1_scores = {}
    for i in range(len(labels)):
        f1_scores[labels[i]] = (2*precesions[labels[i]]*recalls[labels[i]])/(precesions[labels[i]]+recalls[labels[i]])
    return f1_scores