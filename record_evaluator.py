import os
import numpy as np
from signal_reader import SignalReader
from Utils.inference import AfibInference


class RecordEvaluator(AfibInference):
    def __init__(self, dest_dir):
        super(AfibInference, self).__init__()

        self._dest_dir = dest_dir

    def evaluate(self, signal_reader: SignalReader):
        signal = signal_reader.read_signal()
        fs = signal_reader.read_fs()

        # TODO:
        signal = self.preprocessSignal(signal, fs)
        afib_mask = self.makeInference(signal)

        result = np.round(np.random.random(size=[signal.shape[0], ]))

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), result)
