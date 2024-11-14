from torch.utils.data import Dataset
import wfdb
import numpy as np
from pathlib import Path


class EcgDatasetCompiler():
    def __init__(self, dst_path: str, fs: int, sample_size: int, afib_thresh: float, filter=None):
        self.dst_path = dst_path
        self.fs = fs
        self.slice_length = sample_size
        self.filter = filter
        self.afib_thresh = afib_thresh

    def compileEcgDataset(self, src_path: str, first_n_sample_idx: int, first_afib_sample_idx: int):
        with open(src_path + "/RECORDS") as file:
            RECORDS = file.read()

        RECORDS = [item for item in RECORDS.split("\n") if item]

        n_sample_idx = first_n_sample_idx
        afib_sample_idx = first_afib_sample_idx

        for recordIdx in RECORDS:
            try:
                record = wfdb.rdrecord(src_path + "/" + recordIdx)
            except:
                print(f"Failed to load record {recordIdx}")
                continue

            annotation = wfdb.rdann(src_path + "/" + recordIdx, 'atr')
            waveform = record.p_signal[:,0]

            if record.fs != self.fs:
                waveform, annotation = wfdb.processing.resample_multichan(
                    waveform, 
                    annotation,
                    record.fs,
                    self.fs
                )

            if self.filter is not None:
                pass # TODO: Filter the waveform
        
            ann_labels = annotation.aux_note
            ann_samples = annotation.sample

            afib_ranges, afib_mask = self.getAfibMask(waveform, ann_samples, ann_labels)

            waveform_slices, mask_slices = self.sliceWaveform(
                waveform,
                afib_mask,
                self.slice_length,
                ann_samples[0]
            )

            n_sample_idx, afib_sample_idx = self.saveToDataset(
                waveform_slices,
                mask_slices,
                n_sample_idx,
                afib_sample_idx
            )

    def getAfibMask(self, waveform, ann_samples, ann_labels):
        afib_mask = np.zeros(waveform.shape[0])
        afib_ranges = []
        for i, label in enumerate(ann_labels):
            if label == '(AFIB' and i != (len(ann_labels) - 1):
                afib_mask[ann_samples[i]:ann_samples[i+1]] = 1
                afib_ranges.append([ann_samples[i], ann_samples[i+1]])
            elif label == '(AFIB':
                afib_mask[ann_samples[i]:] = 1
                afib_ranges.append([ann_samples[i], (waveform.shape[0]-1)])

        afib_ranges = np.array(afib_ranges)

        return afib_ranges, afib_mask
    
    def sliceWaveform(self, waveform, afib_mask, window, start_sample):
        n_slices = waveform[start_sample:].shape[0] // window
        waveform_slices = []
        mask_slices = []
        for i in range(0, n_slices):
            idx_start = start_sample + i*window
            idx_end = idx_start + window
            waveform_slices.append(waveform[idx_start:idx_end])
            mask_slices.append(afib_mask[idx_start:idx_end])

        return waveform_slices, mask_slices
    
    def saveToDataset(self, waveform_slices, mask_slices, first_n_sample_idx, first_afib_sample_idx):
        Path(self.dst_path + "/afib_samples").mkdir(parents=True, exist_ok=True)
        Path(self.dst_path + "/n_samples").mkdir(parents=True, exist_ok=True)

        i = first_n_sample_idx
        j = first_afib_sample_idx
        for waveform_slice, mask_slice in zip(waveform_slices, mask_slices):
            afib_ratio = np.sum(mask_slice) / mask_slice.size
            if afib_ratio > self.afib_thresh:
                with open(f"{self.dst_path}/afib_samples/afib{j}.npy","wb") as f:
                    np.save(f, waveform_slice) # TODO: Change file format to .npz
                j += 1
            else:
                with open(f"{self.dst_path}/n_samples/n{i}.npy","wb") as f:
                    np.save(f, waveform_slice)
                i += 1
        
        return i, j 


class EcgDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class ToSpectrogram():
    def __init__(self):
        pass

    def __call__(self):
        pass


class BandPassFilter():
    def __init__(self):
        pass

    def __call__(self):
        pass