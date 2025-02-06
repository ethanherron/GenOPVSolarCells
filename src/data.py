import os, torch
import numpy as np
from torch.utils import data

class MicrostructureDataset(data.Dataset):
    """
    Class to read the numpy dataset for the microstructure
    """
    def __init__(self, data_path):
        self.microstructure = np.load(data_path)['arr_0']

    def __len__(self):
        return self.microstructure.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.microstructure[index])
