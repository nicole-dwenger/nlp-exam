"""
UTILS FOR LEXICON EVALUATION USING SENTIMENT CLASSIFICATION TASKS

This script contins utility functions for sentiment classification tasks, used in 
../scr/lexicon_evaluation.py

Many of the functions are taken and adopted from:
https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks
"""

# --- DEPENDENCIES ---

import sys, os
import numpy as np
from tabulate import tabulate

# --- FUNCTIONS ---

def sentiment_score_to_label(score):
    """
    Function to turn continuous scores into classes of positive/neutral/negative
    """
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'

def sentiment_score_to_label_asent(score, pos_threshold, neg_threshold):
    """
    Function to turn continuous scores between -1 and 1 into classes of positive/neutral/negative
    - This should be used for asent to classify document compound scores estiamted by asent
    """
    if score > pos_threshold:
        return 'positiv'
    if score < neg_threshold:
        return 'negativ'
    else:
        return 'neutral'
    
def f1_class(k, true, pred):
    """
    Function adopted from https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks
    - Calculates values for F1 report of sentiment classifiction task
    """
    tp = np.sum(np.logical_and(pred == k, true == k))
    fp = np.sum(np.logical_and(pred == k, true != k))
    fn = np.sum(np.logical_and(pred != k, true == k))
    if tp == 0:
        return 0, 0, 0, 0, 0, 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return tp, fp, fn, precision, recall, f1


def f1_report(true, pred, modelname="", dataname="", word_level=False, bio=False, output_file=""):
    """
    Function adopted from https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks
    - Creates pretty F1 report based on true and predicted values
    """
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
    
    # saves f1 report to output fil
    with open(output_file, 'a') as f:
        print(tabulate(data_b, headers=headers_b, colalign=aligns_b), '\n', file=f)
        f.close()
