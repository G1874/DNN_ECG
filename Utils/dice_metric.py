import numpy as np
from matplotlib import pyplot as plt


def drawSignal(signal, afib_ranges=None, eval_mask=None):
    _, ax = plt.subplots()
    ax.plot(signal)

    if type(eval_mask) != type(None):
        ax.plot(eval_mask)

    if type(afib_ranges) != type(None):
        for x1, x2 in afib_ranges:
            ax.axvspan(x1, x2, alpha=0.5, color='red')

    plt.show()

def confusionMatrix(ref_mask, eval_mask):
    TP = np.sum((ref_mask == 1) & (eval_mask == 1))
    TN = np.sum((ref_mask == 0) & (eval_mask == 0))
    FP = np.sum((ref_mask == 0) & (eval_mask == 1))
    FN = np.sum((ref_mask == 1) & (eval_mask == 0))

    return TP, TN, FP, FN

def diceMetric(TP, FP, FN):
    return 2*TP / (2*TP + FP + FN)