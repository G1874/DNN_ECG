import numpy as np
import pandas as pd
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from scipy.signal import butter, filtfilt
import random
import wfdb
from wfdb import processing
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import os
import shutil
import csv
import json


class EcgDatasetCompiler():
    def __init__(self, dst_path: str, fs: int, sample_size: int, afib_thresh: float, filter=None, transform=None):
        self.dst_path = dst_path
        self.fs = fs
        self.slice_length = sample_size
        self.filter = filter
        self.afib_thresh = afib_thresh
        self.transform = transform

    def compileEcgDataset(self, src_path: str):
        dir_path = os.path.dirname(src_path)
        with open(src_path) as file:
            RECORDS = file.read()

        RECORDS = [item for item in RECORDS.split("\n") if item]
        sample_listing = []
        num_n_samples = 0
        num_afib_samples = 0
        info_dict = dict()

        print("Compiling dataset")
        for record_name in tqdm(RECORDS):
            record_idx = os.path.basename(record_name)

            try:
                record = wfdb.rdrecord(os.path.join(dir_path, record_name))
            except:
                print(f" Failed to load record {record_idx}")
                continue

            annotation = wfdb.rdann(os.path.join(dir_path, record_name), 'atr')
            waveform = record.p_signal[:,0]

            if record.fs != self.fs:
                waveform, annotation = processing.resample_singlechan(
                    waveform, 
                    annotation,
                    record.fs,
                    self.fs
                )

            waveform = processing.normalize_bound(waveform, -1, 1)

            if self.filter is not None:
                waveform = self.filter(waveform)
        
            ann_labels = annotation.aux_note
            ann_samples = annotation.sample

            afib_ranges, afib_mask = self.getAfibMask(waveform, ann_samples, ann_labels)

            waveform_slices, mask_slices = self.sliceWaveform(
                waveform,
                afib_mask,
                self.slice_length,
                ann_samples[0]
            )

            num_n, num_afib = self.saveToDataset(
                waveform_slices,
                mask_slices,
                record_idx,
                sample_listing
            )

            info_dict[f"num_n_{record_idx}"] = num_n
            info_dict[f"num_afib_{record_idx}"] = num_afib
            num_n_samples += num_n
            num_afib_samples += num_afib
        
        print("Done")

        with open(self.dst_path + "/sample_listing.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_listing)

        info_dict["num_n_total"] = num_n_samples
        info_dict["num_afib_total"] = num_afib_samples
        with open(self.dst_path + "/info.txt", 'w') as f:
            json.dump(info_dict, f)

    def getAfibMask(self, waveform, ann_samples, ann_labels):
        samples = []
        labels = []
        for sample, label in zip(ann_samples, ann_labels):
            if label != '':
                if label[0] == '(':
                    samples.append(sample)
                    labels.append(label)
        
        samples = np.array(samples)
        
        afib_mask = np.zeros(waveform.shape[0])
        afib_ranges = []
        for i, label in enumerate(labels):
            if label == '(AFIB' and i != (len(labels) - 1):
                afib_mask[samples[i]:samples[i+1]] = 1
                afib_ranges.append([samples[i], samples[i+1]])
            elif label == '(AFIB':
                afib_mask[samples[i]:] = 1
                afib_ranges.append([samples[i], (waveform.shape[0]-1)])

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
    
    def saveToDataset(self, waveform_slices, mask_slices, record_idx, sample_listing):
        Path(self.dst_path + "/afib_samples").mkdir(parents=True, exist_ok=True)
        Path(self.dst_path + "/n_samples").mkdir(parents=True, exist_ok=True)

        afib_dict = dict()
        n_dict = dict()
        sample_list = [f"record_{record_idx}"]

        i = 0; j = 0
        for waveform_slice, mask_slice in zip(waveform_slices, mask_slices):
            afib_ratio = np.sum(mask_slice) / mask_slice.size
            if afib_ratio > self.afib_thresh:
                j += 1
                afib_dict[f"afib{j}"] = waveform_slice
                sample_list.append(f"afib{j}")
            else:
                i += 1
                n_dict[f"n{i}"] = waveform_slice
                sample_list.append(f"n{i}")

        np.savez(f"{self.dst_path}/afib_samples/afib_samples_record_{record_idx}.npz", **afib_dict)
        np.savez(f"{self.dst_path}/n_samples/n_samples_record_{record_idx}.npz", **n_dict)
        sample_listing.append(sample_list)

        return i, j

    def restructureDataset(self, delete_files=False, max_file_samples=10000):
        Path(self.dst_path + "/dataset").mkdir(parents=True, exist_ok=True)

        with open(self.dst_path + "/info.txt", "r") as f:
            info_dict = json.load(f)

        num_n_samples = info_dict["num_n_total"]
        num_afib_samples = info_dict["num_afib_total"]

        annotation = []
        dataset = dict()
        sample_idx = 0
        file_idx = 0

        if num_afib_samples < num_n_samples:
            minority_set = "/afib_samples/"
            majority_set = "/n_samples/"
            random_indices = random.sample(range(num_n_samples), num_afib_samples)
        else:
            minority_set = "/n_samples/"
            majority_set = "/afib_samples/"
            random_indices = random.sample(range(num_afib_samples), num_n_samples)

        random_indices.sort()

        lower_bound = 0
        upper_bound = 0

        print("Restructuring the dataset")
        for file1, file2 in zip(tqdm(os.listdir(self.dst_path + minority_set)),
                                     os.listdir(self.dst_path + majority_set)):
            
            record = dict(np.load(self.dst_path + minority_set + file1))
            key_list = list(record.keys())

            for key in key_list:
                sample = record[key]
                
                if self.transform:
                    sample = self.transform(sample)
                
                dataset[f"sample{sample_idx}"] = sample
                annotation.append([file_idx,sample_idx,int(key[0]!='n')])
                
                sample_idx += 1
                if len(dataset) == max_file_samples:
                    np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **dataset)
                    file_idx += 1
                    dataset.clear()

            record = dict(np.load(self.dst_path + majority_set + file2))
            key_list = list(record.keys())

            upper_bound += len(key_list)
            record_indices = list(filter(lambda x: x >= lower_bound and x < upper_bound, random_indices))
            key_list = [key_list[i-lower_bound] for i in record_indices]
            lower_bound = upper_bound
            
            for key in key_list:
                sample = record[key]
                
                if self.transform:
                    sample = self.transform(sample)
                
                dataset[f"sample{sample_idx}"] = sample
                annotation.append([file_idx,sample_idx,int(key[0]!='n')])
                
                sample_idx += 1
                if len(dataset) == max_file_samples:
                    np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **dataset)
                    file_idx += 1
                    dataset.clear()

        if dataset:
            np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **dataset)

        with open(self.dst_path + "/dataset/annotation.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["FileIdx","SampleIdx","Label"])
            writer.writerows(annotation)

        if delete_files:
            shutil.rmtree(self.dst_path + "/n_samples")
            shutil.rmtree(self.dst_path + "/afib_samples")
            os.remove(self.dst_path + "/info.txt")
            os.remove(self.dst_path + "/sample_listing.csv")


class EcgDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = root_dir
        self.transform = transform
        self.annotation = np.array(pd.read_csv(root_dir + "/annotation.csv"))
        self.labels = self.annotation[:,2]
        self.file_idx = self.annotation[:,0]
        file_names = [file for file in os.listdir(self.root) if file.startswith("samples")]
        file_names.sort(key=lambda x: int((x[7:])[:-4]))
        self.sample_files = [np.load(os.path.join(self.root,file_name))
                             for file_name in file_names]

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        sample_file = self.sample_files[self.file_idx[idx]]
        sample = sample_file.get(f"sample{idx}")
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
        

class ToSpectrogram():
    def __init__(self, configuration: dict):
        self.w_N = configuration["window_size"]
        self.hop = configuration["stride"]
        self.fs = configuration["fs"]

    def __call__(self, sample):
        w = hamming(self.w_N, True)
        STFT = ShortTimeFFT(w, self.hop, self.fs)
        Sx2 = STFT.spectrogram(sample)
        Sx_dB = 10 * np.log10(np.clip(Sx2, 1e-4, None))
        # Sx_dB = 10 * np.log10(Sx2)
        Sx_dB = Sx_dB[:,(-STFT.p_min):(STFT.p_max(sample.shape[0]))]
        Sx_dB = np.flip(Sx_dB, axis=0)

        return Sx_dB.copy()


class ToTensor():
    def __call__(self, sample):
        min_val = sample.min()
        max_val = sample.max()
        sample = (sample - min_val) / (max_val - min_val + 1e-8)

        sample = torch.tensor(sample.astype(np.float32))
        return sample.reshape((1,sample.shape[0],sample.shape[1]))


class BandPassFilter():
    def __init__(self, configuration: dict):
        self.lowcut = configuration["lowcut"]
        self.highcut = configuration["highcut"]
        self.fs = configuration["fs"]
        self.order = configuration["order"]
        
        self.b, self.a = butter(self.order, [self.lowcut / (0.5 * self.fs), self.highcut / (0.5 * self.fs)], btype='band')

    def __call__(self, signal):
        return filtfilt(self.b, self.a, signal)