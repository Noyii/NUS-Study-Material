import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def standard_trainer(model, train_dataset):
    total_loss, total_err = 0., 0.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    opt = optim.SGD(model.parameters(), lr=1e-1)

    for X, y in train_loader:
        y = ((y + 1) / 2).reshape(-1, 1) # convert the label from {-1,1} to {0,1} for using cross entropy loss
        yp = model(X.float())
        loss = nn.BCEWithLogitsLoss()(yp, y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(train_loader.dataset), total_loss / len(train_loader.dataset)
