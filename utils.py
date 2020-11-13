import torch
import numpy as np
from torch.utils.data import Dataset
from query_probability import load_ucr
import warnings

warnings.filterwarnings('ignore')


class UcrDataset(Dataset):
    def __init__(self, txt_file, channel_last, normalize):
        '''
        :param txt_file: path of file
        :param channel_last
        '''
        # self.data = np.loadtxt(txt_file)
        self.data = load_ucr(txt_file, normalize)
        self.channel_last = channel_last
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])
        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if not self.channel_last:
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:
            return self.data[idx, 1:, :], self.data[idx, 0, :]

    def get_seq_len(self):
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1


class AdvDataset(Dataset):
    def __init__(self, txt_file):
        self.data = np.loadtxt(txt_file)
        self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 2:, :], self.data[idx, 1, :]

    def get_seq_len(self):
        return self.data.shape[1] - 2


def UCR_dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return data_loader
