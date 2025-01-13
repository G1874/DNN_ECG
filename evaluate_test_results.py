import os
import numpy as np
from matplotlib import pyplot as plt
from Utils.preprocessing import EcgDatasetCompiler
import wfdb
from Utils.dice_metric import drawSignal, diceMetric, confusionMatrix


database_path = './Data/mit_bih_atrial_fibrillation_database/files'

def main(save_result_dir, records):
    ecgDatasetCompiler = EcgDatasetCompiler(None, None, None, None)
    
    cum_TP = 0
    cum_FP = 0
    cum_FN = 0

    for record_file_name in records:
        record = wfdb.rdrecord(os.path.join(database_path, record_file_name))
        annotation = wfdb.rdann(os.path.join(database_path, record_file_name), 'atr')

        signal = record.p_signal[:,0]
        ann_labels = annotation.aux_note
        ann_samples = annotation.sample

        afib_ranges, ref_mask = ecgDatasetCompiler.getAfibMask(signal, ann_samples, ann_labels)

        eval_mask = np.load(os.path.join(save_result_dir, f"{record_file_name}.npy"))

        drawSignal(signal, afib_ranges, eval_mask)

        TP, _, FP, FN = confusionMatrix(ref_mask, eval_mask)

        print(f"Accuracy for record {record_file_name}: {diceMetric(TP,FP,FN)}")

        cum_TP += TP
        cum_FP += FP
        cum_FN += FN
    
    print(f"Overall accuracy: {diceMetric(cum_TP,cum_FP,cum_FN)}")

if __name__ == "__main__":
    save_result_dir = "./Testing/output"
    eval_files_list = "./Testing/input/RECORDS"

    with open(eval_files_list, 'r') as f:
        records = f.read().split('\n')

    main(save_result_dir, records)