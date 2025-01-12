from preprocessing import EcgDatasetCompiler, ToSpectrogram
from torchvision import transforms

#
# Prepare dataset for training.
# For now works only for mit_bih_atrial_fibrillation_database.
#

sptectrogram_config = {
    "window_size": 128,
    "stride": 128//8,
    "fs": 250
}

transform = transforms.Compose([ToSpectrogram(sptectrogram_config)])

ecgDatasetCompiler = EcgDatasetCompiler(
    "Data/training_data",   # Output path of the compiled dataset.
    250,                    # Sampling rate in Hz.
    5*250,                  # Size of a single slice of the record.
    0.5,                    # Threshold of afib ratio in a slice to classify it as afib
    filter=None,
    transform=transform
)

ecgDatasetCompiler.compileEcgDataset("./Data/mit_bih_atrial_fibrillation_database/files")
ecgDatasetCompiler.restructureDataset(delete_files=True, max_file_samples=10000)