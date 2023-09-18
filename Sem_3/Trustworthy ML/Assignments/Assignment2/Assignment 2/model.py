
from sklearn.metrics import hinge_loss, zero_one_loss,accuracy_score,log_loss
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from defense import standard_trainer
import numpy as np
'''
This file contains the implementation of the models. Students don't need to modify it.
'''

class Model:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def predict(self, x_test):
        if self.model_name == 'svm':
            return self.model.predict(x_test)
        elif self.model_name == 'nn' or self.model_name == 'lr':
            x_test = torch.tensor(x_test)
            pred = torch.sigmoid(self.model(x_test.float()))
            return (pred > 0.5).numpy().astype(int)

    def predict_proba(self,x_test):
        if self.model_name == 'svm':
            return self.model.decision_function(x_test)
        elif self.model_name == 'nn' or self.model_name == 'lr':
            x_test = torch.tensor(x_test).float()
            return torch.sigmoid(self.model(x_test)).detach().numpy().astype(np.float64)


    def score(self, test_dataset):
        if self.model_name == 'svm':
            return hinge_loss(test_dataset.Y, self.predict_proba(test_dataset.X)),accuracy_score(test_dataset.Y, self.predict(test_dataset.X))
        elif self.model_name == 'nn' or self.model_name == 'lr':
            return log_loss((test_dataset.Y + 1) / 2, self.predict_proba(test_dataset.X)), accuracy_score(
                (test_dataset.Y + 1) / 2, self.predict(test_dataset.X))

    def individual_loss(self, test_dataset):
        num_samples = test_dataset.Y.shape[0]
        loss = np.zeros((num_samples, 1))

        if self.model_name == 'svm':
            pred = self.model.decision_function(test_dataset.X)
            for i in range(len(test_dataset)):
                loss[i] = hinge_loss([test_dataset.Y[i]], [pred[i]])
        elif self.model_name == 'nn' or self.model_name == 'lr':
            pred = self.predict_proba(test_dataset.X)
            target = (test_dataset.Y + 1) / 2
            for i in range(len(test_dataset)):
                loss[i] = log_loss([target[i]], [pred[i]],labels=[0,1])
        return loss


class Undefended_Model(Model):
    def train(self, train_dataset):
        if self.model_name =='svm':
            self.model.fit(train_dataset.X, train_dataset.Y)
        elif self.model_name == 'nn' or self.model_name == 'lr':
            for i in range(50):
                standard_trainer(self.model, train_dataset)





