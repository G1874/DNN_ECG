import wfdb
from Utils.preprocessing import EcgDatasetCompiler
import os
from matplotlib import pyplot as plt

def show_record(record_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    wfdb.plot_wfdb(record=record, annotation=annotation)

def show_signal(record_path):
    ecgDatasetCompiler = EcgDatasetCompiler(None, None, None, None)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:,0]
    ann_labels = annotation.aux_note
    ann_samples = annotation.sample

    afib_ranges, ref_mask = ecgDatasetCompiler.getAfibMask(signal, ann_samples, ann_labels)

    _, ax = plt.subplots()
    ax.plot(signal)
    # ax.plot(eval_mask)
    for x1, x2 in afib_ranges:
        ax.axvspan(x1, x2, alpha=0.5, color='red')
    plt.show()