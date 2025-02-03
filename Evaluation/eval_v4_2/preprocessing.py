import numpy as np
import pandas as pd
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from scipy.signal import butter, filtfilt
import torch


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