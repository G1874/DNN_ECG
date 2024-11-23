from torch.utils.data import Dataset
import wfdb
import numpy as np
from pathlib import Path
import csv
import json
import os
import random
import shutil
from tqdm import tqdm


class EcgDatasetCompiler():
    def __init__(self, dst_path: str, fs: int, sample_size: int, afib_thresh: float, filter=None):
        self.dst_path = dst_path
        self.fs = fs
        self.slice_length = sample_size
        self.filter = filter
        self.afib_thresh = afib_thresh

    def compileEcgDataset(self, src_path: str):
        with open(src_path + "/RECORDS") as file:
            RECORDS = file.read()

        RECORDS = [item for item in RECORDS.split("\n") if item]
        sample_listing = []
        num_n_samples = 0
        num_afib_samples = 0
        info_dict = dict()

        print("Compiling dataset")
        for record_idx in tqdm(RECORDS):
            try:
                record = wfdb.rdrecord(src_path + "/" + record_idx)
            except:
                print(f" Failed to load record {record_idx}")
                continue

            annotation = wfdb.rdann(src_path + "/" + record_idx, 'atr')
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

    def restructureDataset(self, deleteFiles=False):
        Path(self.dst_path + "/dataset").mkdir(parents=True, exist_ok=True)

        with open(self.dst_path + "/info.txt", "r") as f:
            info_dict = json.load(f)

        num_n_samples = info_dict["num_n_total"]
        num_afib_samples = info_dict["num_afib_total"]

        annotation = []
        train_dataset = dict()
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

        print("Restructuring the dataset")
        print(minority_set)
        for file in tqdm(os.listdir(self.dst_path + minority_set)):
            record = dict(np.load(self.dst_path + minority_set + file))
            for key in record.keys():
                train_dataset[f"sample{sample_idx}"] = record[key]
                annotation.append([file_idx,sample_idx,1])
                sample_idx += 1
                if len(train_dataset) == 10000:
                    np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **train_dataset)
                    file_idx += 1
                    train_dataset.clear()
        print("Done")

        lower_bound = 0
        upper_bound = 0

        print(majority_set)
        for file in tqdm(os.listdir(self.dst_path + majority_set)):
            record = dict(np.load(self.dst_path + majority_set + file))
            key_list = list(record.keys())

            upper_bound += len(key_list)
            record_indices = list(filter(lambda x: x >= lower_bound and x < upper_bound, random_indices))
            key_list = [key_list[i-lower_bound] for i in record_indices]
            lower_bound = upper_bound
            
            for key in key_list:
                train_dataset[f"sample{sample_idx}"] = record[key]
                annotation.append([file_idx,sample_idx,0])
                sample_idx += 1
                if len(train_dataset) == 10000:
                    np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **train_dataset)
                    file_idx += 1
                    train_dataset.clear()
        print("Done")

        if train_dataset:
            np.savez(f"{self.dst_path}/dataset/samples{file_idx}.npz", **train_dataset)

        with open(self.dst_path + "/dataset/annotation.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(annotation)

        if deleteFiles:
            shutil.rmtree(self.dst_path + "/n_samples")
            shutil.rmtree(self.dst_path + "/afib_samples")
            os.remove(self.dst_path + "/info.txt")
            os.remove(self.dst_path + "/sample_listing.csv")


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