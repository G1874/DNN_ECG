import numpy as np
import torch
from torchvision import transforms
from wfdb import processing
from Utils.preprocessing import ToSpectrogram, ToTensor, BandPassFilter
from Network.afib_detector_v1_0 import AfibDetector


class AfibInference():
    def __init__(self):
        self.model_path = "./Models/afib_detector_v1_0_4.pt"
        self.fs = 250 # Sampling frequency
        self.stride = 1250 // 2
        self.window_size = 1250

        filter_config = {
            "lowcut": 0.5,
            "highcut": 50,
            "fs": self.fs,
            "order": 5
        }
        self.filter = BandPassFilter(filter_config)

        sptectrogram_config = {
            "window_size": 128,
            "stride": 128//8,
            "fs": self.fs
        }

        self.transform = transforms.Compose([
            ToSpectrogram(sptectrogram_config),
            ToTensor(),
            transforms.Resize([64, 64])
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocessSignal(self, signal, fs):
        signal = signal[:,0]
        
        if fs != self.fs:
            signal = processing.resample_sig(
                signal,
                fs,             # Original frequency
                self.fs         # Frequency target
            )

        if self.filter is not None:
            signal = self.filter(signal)

        return signal
    
    def makeInference(self, signal, fs):
        length = signal.shape[0]

        output_mask = np.zeros(length, dtype=np.float32)
        overlap = np.zeros(length, dtype=np.float32)

        net = self.loadNetwork()

        for step in range(0, length//self.stride):
            idx = step * self.stride

            if (idx + self.window_size) >= length:
                idx = (length - 1) - self.window_size
            
            overlap[idx:(idx+self.window_size)] += 1.0
            slice = signal[idx:(idx+self.window_size)]
            
            prediction = self.classifySlice(net, slice)

            output_vector = np.ones(self.window_size) * prediction.numpy()
            p_output_vector = output_mask[idx:(idx+self.window_size)]
            overlap_vector = overlap[idx:(idx+self.window_size)]

            output_mask[idx:(idx+self.window_size)] = ((overlap_vector - 1) * p_output_vector + output_vector) / overlap_vector

        if fs != self.fs:
            output_mask = processing.resample_sig(
                output_mask,
                self.fs,        # Original frequency
                fs              # Frequency target
            )

        return np.where(output_mask > 0.5, 1.0, 0.0) 

    def classifySlice(self, net, slice):
        input = self.transform(slice)
        input = input.reshape((1,input.shape[0],input.shape[1],input.shape[2]))

        with torch.no_grad():
            output = net(input)
        prediction = torch.argmax(output)

        return prediction

    def loadNetwork(self):
        net = AfibDetector()
        net.load_state_dict(torch.load(self.model_path,
                                       weights_only=True,
                                       map_location=torch.device('cpu')))
        net.eval()

        return net