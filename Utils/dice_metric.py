import numpy as np
from matplotlib import pyplot as plt

def drawSignal(signal, afib_ranges, eval_mask):
    eval_range_start = []
    eval_range_end = []
    for i, element in enumerate(eval_mask):
        if i == 0:
            if element == 1:
                eval_range_start.append(i)
            continue
        if (element == 1) and (eval_mask[i-1] == 0):
            eval_range_start.append(i)

        if i == (eval_mask.size - 1) and element == 1:
            eval_range_end.append(i)
            continue
        if (element == 0) and (eval_mask[i-1] == 1):
            eval_range_end.append(i)

    _, ax = plt.subplots()
    ax.plot(signal)
    # ax.plot(eval_mask)
    for x1, x2 in afib_ranges:
        ax.axvspan(x1, x2, alpha=0.5, color='red')
    for x1, x2 in zip(eval_range_start, eval_range_end):
        ax.axvspan(x1, x2, alpha=0.5, color='blue')
    plt.show()

def confusionMatrix(ref_mask, eval_mask):
    TP = np.sum((ref_mask == 1) & (eval_mask == 1))
    TN = np.sum((ref_mask == 0) & (eval_mask == 0))
    FP = np.sum((ref_mask == 0) & (eval_mask == 1))
    FN = np.sum((ref_mask == 1) & (eval_mask == 0))

    return TP, TN, FP, FN

def diceMetric(TP, FP, FN):
    return 2*TP / (2*TP + FP + FN)