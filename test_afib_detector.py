import os
import time
from signal_reader import SignalReader
from record_evaluator import RecordEvaluator


database_path = './Data/mit_bih_atrial_fibrillation_database/files'

def main(save_result_dir, records):
    print(f'no of records: {len(records)}')
    
    eval_start = time.time()
    
    for record_file_name in records:
        print(f'processing record: {record_file_name}')
        start = time.time()
        
        signal_reader = SignalReader(os.path.join(database_path, record_file_name))
        record_eval = RecordEvaluator(save_result_dir)

        record_eval.evaluate(signal_reader)
        
        print(f'took: {time.time() - start} seconds')
    
    print(f'full eval took {time.time() - eval_start} seconds')

if __name__ == "__main__":
    save_result_dir = "./Testing/output"
    eval_files_list = "./Testing/input/RECORDS"

    with open(eval_files_list, 'r') as f:
        records = f.read().split('\n')
    
    main(save_result_dir, records)