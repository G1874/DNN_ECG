import os
import numpy as np
from matplotlib import pyplot as plt
from Utils.preprocessing import EcgDatasetCompiler
import wfdb
from Utils.dice_metric import drawSignal, diceMetric, confusionMatrix


database_path = './Data'

def main(save_result_dir, records):
    ecgDatasetCompiler = EcgDatasetCompiler(None, None, None, None)
    
    cum_TP = 0
    cum_TN = 0
    cum_FP = 0
    cum_FN = 0

    for record_file_name in records:
        record = wfdb.rdrecord(os.path.join(database_path, record_file_name))
        annotation = wfdb.rdann(os.path.join(database_path, record_file_name), 'atr')

        signal = record.p_signal[:,0]
        ann_labels = annotation.aux_note
        ann_samples = annotation.sample

        afib_ranges, ref_mask = ecgDatasetCompiler.getAfibMask(signal, ann_samples, ann_labels)

        record_idx = os.path.basename(record_file_name)
        eval_mask = np.load(os.path.join(save_result_dir, f"{record_idx}.npy"))

        drawSignal(signal, afib_ranges, eval_mask)

        TP, TN, FP, FN = confusionMatrix(ref_mask, eval_mask)

        print(f"Metrics for the record: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print(f"Accuracy for record {record_file_name}: {diceMetric(TP,FP,FN)}")

        cum_TP += TP
        cum_TN += TN
        cum_FP += FP
        cum_FN += FN
    
    print(f"Overall metrics: TP={cum_TP}, TN={cum_TN}, FP={cum_FP}, FN={cum_FN}")
    print(f"Overall accuracy: {diceMetric(cum_TP,cum_FP,cum_FN)}")

if __name__ == "__main__":
    save_result_dir = "./Testing/output"
    eval_files_list = "./Testing/RECORDS"

    with open(eval_files_list, 'r') as f:
        records = f.read()

    records = [item for item in records.split("\n") if item]

    main(save_result_dir, records)