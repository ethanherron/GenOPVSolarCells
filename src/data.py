import os, torch
import numpy as np
from torch.utils import data
from torchvision import transforms


class TopoDataset2D(data.Dataset):
    'PyTorch dataset for Topology Optimization'

    def __init__(self, data_path, mode='train'):

        data_path = os.path.join(data_path, 'generator_data')
        if mode == 'train':
            self.data_path = os.path.join(data_path, 'train')
        elif mode == 'validation':
            self.data_path = os.path.join(data_path, 'validation')

        self.final_D = np.load(os.path.join(self.data_path, 'final_D.npz'))['arr_0']

    def __len__(self):
        'Denotes the total number of samples'
        return self.final_D.shape[0]
    
    def log_normalization(self, x):
        x = torch.clamp(x, min = 1e-22, max = None)
        x = (22 + torch.log10(torch.clamp(x/torch.max(x), 1e-22, 1.0)))/22.0
        return x

    def __getitem__(self, index):
        'Generates one sample of data'
        return torch.FloatTensor(self.final_D[index])



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
