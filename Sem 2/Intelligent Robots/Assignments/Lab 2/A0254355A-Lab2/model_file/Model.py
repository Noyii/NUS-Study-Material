import os
import pandas as pd
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

# Change this to cuda 
gpu_name = "mps"


class MyModel(nn.Module):

    def __init__(self, num_bins=5):
        super().__init__()
        self.num_bins = num_bins

        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_bins)

        # Build a FC heads, taking both the image features and the intention as input
        self.fc = nn.Sequential(nn.Linear(in_features=num_bins+3, out_features=num_bins))
        print(f'A not so simple learner.')


    def forward(self, image, intention):
        # Map images to feature vectors
        feature = self.model_ft(image).flatten(1)

        # Cast intention to one-hot encoding 
        intention = intention.unsqueeze(1)
        onehot_intention = torch.zeros(intention.shape[0], 3, device=intention.device).scatter_(1, intention, 1)
        
        # Predict control
        control = self.fc(torch.cat((feature, onehot_intention), dim=1)).view(-1, self.num_bins)

        # Return control as a categorical distribution
        return control


def get_lr(optimizer):
    '''
    Get the current learning rate
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']


def topk_accuracy(k, outputs, targets):
    """
    Compute top k accuracy
    """
    batch_size = targets.size(0)

    _, pred = outputs.topk(k, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.type(torch.FloatTensor).sum().item()

    return n_correct_elems / batch_size


def read_image(path):
    return Image.open(path)


class AverageMeter(object):
    """
    A utility class to compute statisitcs of losses and accuracies
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MyDataset(Dataset):
    
    INTENTION_MAPPING = {'forward': 0, 'left': 1, 'right': 2}
    MAX_VELOCITY = 0.7
    MIN_VELOCITY = -0.7

    def __init__(self, is_train=True, num_bins=5):
        self.bin_size = (self.MAX_VELOCITY - self.MIN_VELOCITY) / num_bins

        self.data_dir = '../BehaviourCloning'
        if is_train:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'train.txt'), sep='  ', engine='python')
        else:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'val.txt'), sep=' ', engine='python')

        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def plot_accuracies(num_epochs, train_accs, val_accs):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.title("Training Summary")
    plt.xlabel("No. of Epochs)")
    plt.plot(range(num_epochs), train_accs, 'g', label='Train Accuracy')
    plt.plot(range(num_epochs), val_accs, 'r', label='Val Accuracy')

    plt.show()

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# def augment_data():
#     return Compose([
#         RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
#         RandomRotation(10)
#     ])


def train():
    batch_size = 64
    num_epochs = 5
    num_workers = 2
    num_bins = 5
    set_seed(42)

    model = MyModel(num_bins)
    model = nn.DataParallel(model.to(gpu_name).float())

    train_set = MyDataset(is_train=True, num_bins=num_bins)
    validation_set = MyDataset(is_train=False, num_bins=num_bins)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.3)
    # augmenter = augment_data()

    train_accs = []
    val_accs = []

    # training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, val_loss = AverageMeter(), AverageMeter()
        train_acc, val_acc = AverageMeter(), AverageMeter()

        # train loop
        model.train()
        for i, (image, intention, label) in enumerate(train_loader):
            image, intention, label = image.to(gpu_name), intention.to(gpu_name), \
                    label.to(gpu_name).view(-1)

            prediction = model(image, intention)

            loss = criterion(prediction, label)
            train_loss.update(loss.item())

            acc = topk_accuracy(2, prediction, label)
            train_acc.update(acc)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 99:
                print(f'time:{datetime.now()} training: iteration {i} / {len(train_loader)}, avg train loss = {train_loss.avg:.4f}, '
                      f'train accuracy {train_acc.avg:.4f}')

        # validation
        model.eval()
        for i, (image, intention, label) in enumerate(validation_loader):
            image, intention, label = image.to(gpu_name), intention.to(gpu_name), \
                label.to(gpu_name).view(-1)
            with torch.no_grad():
                prediction = model(image, intention)

                loss = criterion(prediction, label)
                val_loss.update(loss.item())

                acc = topk_accuracy(2, prediction, label)
                val_acc.update(acc)

            if i % 100 == 99:
                print(f'time:{datetime.now()} validation: iteration {i} / {len(validation_loader)}, avg val loss = {val_loss.avg:.4f}, '
                      f'val accuracy {val_acc.avg:.4f}')

        # epoch summary
        print(f'Epoch {epoch}, train error {train_loss.avg:.4f}, val error {val_loss.avg:.4f}. '
              f'Train acc = {train_acc.avg:.4f}, val acc = {val_acc.avg:.4f}. '
              f'Time cost {(time.time() - start_time) / 60:.2f} min.\n')
        print("--------------------------------------------------------")

        train_accs.append(train_acc.avg)
        val_accs.append(val_acc.avg)

        # lr scheduler
        scheduler.step(val_loss.avg)

        # checkpoint
        model_name = 'adam_resnet_e'
        path = 'Checkpoints/' + model_name + str(epoch) + '.pt'
        torch.save(model.state_dict(), path)

    # Visualize
    plot_accuracies(num_epochs, train_accs, val_accs)


if __name__ == "__main__":
    train()