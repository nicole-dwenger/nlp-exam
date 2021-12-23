"""
Utils for Sentiment Classification Task
Taken from https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks
"""

import sys, os
import numpy as np
from tabulate import tabulate

def sentiment_score_to_label(score):
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'

def sentiment_score_to_label_asent(score):
    # the threshold of 0.4 is fitted on a manually annotated twitter corpus for sentiment on 1327 examples
    if score > 0.8:
        return 'positiv'
    if score < -0.2:
        return 'negativ'
    else:
        return 'neutral'
    
def f1_class(k, true, pred):
    tp = np.sum(np.logical_and(pred == k, true == k))

    fp = np.sum(np.logical_and(pred == k, true != k))
    fn = np.sum(np.logical_and(pred != k, true == k))
    if tp == 0:
        return 0, 0, 0, 0, 0, 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return tp, fp, fn, precision, recall, f1


def f1_report(true, pred, modelname="", dataname="", word_level=False, bio=False):

    if bio:
        return classification_report(true, pred, digits=4)

    if word_level:
        true = [tag for sent in true for tag in sent]
        pred = [tag for sent in pred for tag in sent]

    data_b = []
    data_a = []
    headers_b = ["{} // {} ".format(modelname, dataname), 'Class', 'Precision', 'Recall', 'F1', 'support']
    headers_a = ['Accuracy', 'Avg-f1', 'Weighted-f1', '', '']
    aligns_b = ['left', 'left', 'center', 'center', 'center']

    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum(true == pred) / len(true)

    n = len(np.unique(true))
    avg = 0
    wei = 0
    for c in np.unique(true):
        _, _, _, precision, recall, f1 = f1_class(c, true, pred)
        avg += f1 / n
        wei += f1 * (np.sum(true == c) / len(true))

        data_b.append(['', c, round(precision, 4), round(recall, 4), round(f1, 4)])
    data_b.append(['', '', '', '', ''])
    data_b.append(headers_a)
    data_b.append([round(acc, 4), round(avg, 4), round(wei, 4), '', ''])
    print(tabulate(data_b, headers=headers_b, colalign=aligns_b), '\n')
    
    #with open(f"{output_path}/f1_report.txt", 'a') as f:
    #    print(tabulate(data_b, headers=headers_b, colalign=aligns_b), '\n', file=f)

    