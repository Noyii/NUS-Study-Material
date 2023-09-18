import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN_mini(nn.Module):
    '''
    This is a base CNN model.
    '''

    def __init__(self, feat_dim=256, pitch_class=13, pitch_octave=5):
        '''
        Definition of network structure.
        '''
        super().__init__()
        self.feat_dim = 256
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class

        '''
        YOUR CODE: the remaining part of the model structure
        '''
        self.conv2d_1 = nn.Conv2d(in_channels=1, kernel_size=3, padding=1, out_channels=16)
        self.conv2d_2 = nn.Conv2d(in_channels=16, kernel_size=3, padding=1, out_channels=32)
        self.conv2d_3 = nn.Conv2d(in_channels=32, kernel_size=3, padding=1, out_channels=64)

        self.bn_1 = nn.BatchNorm2d(num_features=16)
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.bn_3 = nn.BatchNorm2d(num_features=64)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))

        self.linear = nn.Linear(in_features=4096, out_features=feat_dim)
        self.on = nn.Linear(in_features=feat_dim, out_features=1)
        self.off = nn.Linear(in_features=feat_dim, out_features=1)
        self.octave = nn.Linear(in_features=feat_dim, out_features=pitch_octave)
        self.pitch = nn.Linear(in_features=feat_dim, out_features=pitch_class)



    def forward(self, x):
        '''
        Compute output from input
        '''
        '''
        YOUR CODE: computing output from input
        '''
        x = x.unsqueeze(1)
        x = self.max_pool(self.relu(self.bn_1(self.conv2d_1(x))))
        x = self.max_pool(self.relu(self.bn_2(self.conv2d_2(x))))
        x = self.relu(self.bn_3(self.conv2d_3(x)))

        x = torch.swapaxes(x, 1, 2).flatten(2)
        x = self.linear(x)

        onset_logits = self.on(x).squeeze()
        offset_logits = self.off(x).squeeze()
        pitch_octave_logits = self.octave(x)
        pitch_class_logits = self.pitch(x)

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits
