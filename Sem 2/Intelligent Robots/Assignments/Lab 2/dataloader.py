import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *

def read_image(path):
    return Image.open(path)


class MyDataset(Dataset):
    
    INTENTION_MAPPING = {'forward': 0, 'left': 1, 'right': 2}
    MAX_VELOCITY = 0.7
    MIN_VELOCITY = -0.7

    def __init__(self, is_train=True, num_bins=5):
        self.bin_size = (self.MAX_VELOCITY - self.MIN_VELOCITY) / num_bins

        self.data_dir = '.'
        if is_train:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'train.txt'), sep='  ')
        else:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'val.txt'), sep=' ')

        self.preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])
        
        print(f'loaded data from {self.data_dir}. dataset size {len(self)}')

    def discretize_control(self, control):
        return int((control - self.MIN_VELOCITY) / self.bin_size)

    def __getitem__(self, idx):
        frame, _, _, angular_velocity, intention = self.data.iloc[idx]
        image = self.preprocess(read_image(os.path.join(self.data_dir, 'images', f'{frame}.jpg')))
        intention = torch.tensor(self.INTENTION_MAPPING[intention])
        label = torch.tensor(self.discretize_control(angular_velocity))

        return image, intention, label

    def __len__(self):
        return len(self.data)