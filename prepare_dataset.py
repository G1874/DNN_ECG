from Utils.preprocessing import EcgDatasetCompiler, ToSpectrogram, BandPassFilter
from torchvision import transforms


#
# Prepare dataset for training.
#

fs = 125                # Sampling rate in Hz.
slice_size = 10*fs      # Size of the slice of the signal that is
                        # fed into the network (10s).

filter_config = {
    "lowcut": 0.5,
    "highcut": 50,
    "fs": fs,
    "order": 7
}

band_pass_filter = BandPassFilter(filter_config)

sptectrogram_config = {
    "window_size": 128,
    "stride": 128//4,
    "fs": fs
}

transform = transforms.Compose([ToSpectrogram(sptectrogram_config)])

ecgDatasetCompiler = EcgDatasetCompiler(
    "Data/training_data",       # Output path of the compiled dataset.
    fs,                         # Sampling rate in Hz.
    slice_size,                 # Size of a single slice of the record.
    0.5,                        # Threshold of afib ratio in a slice to classify it as afib.
    filter=band_pass_filter,    # Band pass filter used on the signal in preprocessing
    transform=transform,        # Transforms used on a slice of the signal in preprocessing.
)

if __name__ == "__main__":
    ecgDatasetCompiler.compileEcgDataset("./Data/RECORDS")
    ecgDatasetCompiler.restructureDataset(delete_files=True, max_file_samples=10000)