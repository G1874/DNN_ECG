from torch.utils.data import Dataset


def compileEcgDataset(path: str):
    pass

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

class BandPassFiltering():
    def __init__(self):
        pass

    def __call__(self):
        pass