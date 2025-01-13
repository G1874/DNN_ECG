import numpy as np
import wfdb
import os


class SignalReader():
    def __init__(self, path):
        self.path = path

    def read_signal(self) -> np.ndarray:
        record = wfdb.rdrecord(self.path)
        return record.p_signal

    def read_fs(self) -> float:
        record = wfdb.rdrecord(self.path)
        return record.fs

    def get_code(self) -> str:
        return "output/" + os.path.basename(self.path)