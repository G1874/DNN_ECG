from preprocessing import EcgDatasetCompiler

#
# Prepare dataset for training.
# For now works only for mit_bih_atrial_fibrillation_database.
#

ecgDatasetCompiler = EcgDatasetCompiler(
    "Data/training_data",   # Output path of the compiled dataset.
    250,                    # Sampling rate in Hz.
    5*250,                  # Size of a single slice of the record.
    0.5                     # Threshold of afib ratio in a slice to classify it as afib
)

ecgDatasetCompiler.compileEcgDataset("./Data/mit_bih_atrial_fibrillation_database/files")
ecgDatasetCompiler.restructureDataset(delete_files=True, max_file_samples=200000)