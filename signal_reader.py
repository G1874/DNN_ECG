import numpy as np
import wfdb
import os


class SignalReader():
    def __init__(self, path):
        self.path = path

    '''
    Funkcja zwraca wczytany sygnal ekg.
    :return: ndarray postaci: n na c, gdzie n to liczba probek, m to ilosc kanalow
    '''
    def read_signal(self) -> np.ndarray:
        record = wfdb.rdrecord(self.path)
        return record.p_signal

    '''
    Funkcja zwraca czestotliwosc probkowania czytanego sygnalu
    :return: czestotliwosc probkowania
    '''
    def read_fs(self) -> float:
        record = wfdb.rdrecord(self.path)
        return record.fs

    '''
    Funkcja zwraca specjalny kod pod nazwa ktore nalezy zapisac wynik ewaluacji w klasie RecordEvaluator
    :return: kod identyfikujacy ewaluacje/nazwa pliku do zapisu ewaluacji
    '''
    def get_code(self) -> str:
        return "./Output/" + os.path.basename(self.path)