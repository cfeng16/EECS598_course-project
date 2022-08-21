import h5py
import numpy as np
import pickle
from torch.utils.data import Dataset

class Trainset(Dataset):
    def __init__(self, train_file, scale_factor):
        super(Trainset, self).__init__()
        self.train_file = train_file
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        with h5py.File(self.train_file, 'r') as file:
            lr = file['lr']
            hr = file['hr']
            lr = np.transpose(lr[index], (2, 0, 1)) / 255.
            hr = np.transpose(hr[index], (2, 0, 1)) / 255.
            hr_height = hr.shape[1]
            hr_width = hr.shape[2]
            #hr = hr[:, :(hr_height-self.scale_factor+1), :(hr_width-self.scale_factor+1)]
            return {'hr':hr, 'lr':lr}

    def __len__(self):
        with h5py.File(self.train_file, 'r') as file:
            return len(file['lr'])


class Valset(Dataset):
    def __init__(self, val_file, scale_factor):
        super(Valset, self).__init__()
        self.val_file = val_file
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        with h5py.File(self.val_file, 'r') as file:
            lr = file['lr']
            hr = file['hr']
            lr = np.transpose(lr[str(index)][:], (2, 0, 1)) / 255.
            hr = np.transpose(hr[str(index)][:], (2, 0, 1)) / 255.
            hr_height = hr.shape[1]
            hr_width = hr.shape[2]
            #hr = hr[:, :(hr_height-self.scale_factor+1), :(hr_width-self.scale_factor+1)]
            return {'hr':hr, 'lr':lr}

    def __len__(self):
        with h5py.File(self.val_file, 'r') as file:
            return len(file['lr'])
        
class Testset(Dataset):
    def __init__(self, test_file, scale_factor):
        super(Testset, self).__init__()
        self.test_file = test_file
        self.scale_factor = scale_factor
        #with open(test_file, 'rb') as file:
        #    self.test_data = pickle.load(file)

    def __getitem__(self, index):
        with h5py.File(self.test_file, 'r') as file:
            hr = file['hr'][str(index)][:]
            hr = np.transpose(hr, (2, 0, 1)) / 255.
            lr = file['lr'][str(index)][:]
            hr_height = hr.shape[1]
            hr_width = hr.shape[2]
            #hr = hr[:, :(hr_height-self.scale_factor+1), :(hr_width-self.scale_factor+1)]
            lr = np.transpose(lr, (2, 0, 1)) / 255.
            return {'hr':hr, 'lr':lr}

    def __len__(self):
       with h5py.File(self.test_file, 'r') as file:
            return len(file['lr'])