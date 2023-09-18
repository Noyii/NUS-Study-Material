import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model, preprocessing, cluster, metrics, svm, model_selection, neighbors
import torch.nn as nn
from torchvision import  transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import torch

class dataset:
    def __init__(self,x,y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __delitem__(self, key):
       self.X = np.delete(self.X, key, axis=0)
       self.Y = np.delete(self.Y, key, axis=0)


def load_model(model_name, dataset):

    if model_name == 'nn' and dataset == 'mnist_17':
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )
    elif model_name == 'nn' and dataset == 'dogfish':
        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )
    elif model_name == 'lr' and dataset == 'mnist_17':
        model = torch.nn.Linear(784, 1)
    elif model_name == 'lr' and dataset == 'dogfish':
        model = torch.nn.Linear(2048, 1)
    elif model_name == 'svm':
        model = svm.LinearSVC(
        tol=1e-7,
        loss='hinge',
        fit_intercept=True,
        random_state=24,
        max_iter=10000)
    return model


def load_dataset(data_name):
    DATA_FOLDER = 'dataset'
    if data_name == 'dogfish':
        dataset_path = os.path.join(DATA_FOLDER)
        train_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_train.npz'))
        X_train = train_f['inception_features_val']
        Y_train = np.array(train_f['labels'] * 2 - 1, dtype=int)

        train_dataset = dataset(X_train,Y_train)

        test_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_test.npz'))
        X_test = test_f['inception_features_val']
        Y_test = np.array(test_f['labels'] * 2 - 1, dtype=int)
        check_orig_data(X_train, Y_train, X_test, Y_test)
        test_dataset = dataset(X_test, Y_test)
        return train_dataset,test_dataset

    elif data_name == 'mnist_17':
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, 'mnist_17_train_test.npz'))
        X_train = f['X_train']
        Y_train = f['Y_train'].reshape(-1)
        X_test = f['X_test']
        Y_test = f['Y_test'].reshape(-1)
        train_dataset = dataset(X_train, Y_train)
        test_dataset = dataset(X_test, Y_test)
        check_orig_data(X_train, Y_train, X_test, Y_test)
        return train_dataset,test_dataset
    else:
        raise ValueError('Wrong dataset name, only support dogfish and mnist_17')



def check_orig_data(X_train, Y_train, X_test, Y_test):
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.max(Y_train) == 1, 'max of Y_train was %s' % np.max(Y_train)
    assert np.min(Y_train) == -1
    assert len(set(Y_train)) == 2
    assert set(Y_train) == set(Y_test)


def combine_datset(datasetA,datasetB):
    X = np.concatenate((datasetA.X,datasetB.X),axis=0)
    Y = np.concatenate((datasetA.Y,datasetB.Y),axis=0)

    return  dataset(X,Y)


def loss_diff(clean_dataset,poison_dataset,model):
    # plot the loss distribution of the clean data and poisoning data on the model
    clean_loss = model.individual_loss(clean_dataset)
    poison_loss = model.individual_loss(poison_dataset)

    # min = np.min([np.min(clean_loss), np.min(poison_loss)])
    # max = np.max([np.max(clean_loss), np.max(poison_loss)])
    bins = np.linspace(0, 1, 30)
    plt.hist(clean_loss, bins, alpha=0.5, label='clean data')
    plt.hist(poison_loss, bins, alpha=0.5, label='poisoning data')
    plt.legend(loc='upper right')
    plt.title('The loss distribution')
    plt.show()


def distance_to_center_diff(clean_dataset,poison_dataset):
    # plot the distribution of the distance to the centroid of the clean data

    class_map = {-1: 0, 1: 1}
    X_clean = clean_dataset.X
    Y_clean = clean_dataset.Y

    X_poison = poison_dataset.X
    Y_poison = poison_dataset.Y

    num_classes = len(set(Y_clean))
    num_features = X_clean.shape[1]

    centroids = np.zeros((num_classes, num_features))
    dis_clean_to_centro = np.zeros((len(clean_dataset), 1))
    dis_poison_to_centro = np.zeros((len(poison_dataset), 1))
    for y in set(Y_clean):
        centroids[class_map[y], :] = np.mean(X_clean[Y_clean == y, :], axis=0)

    for i in range(len(clean_dataset)):
        dis_clean_to_centro[i] = np.linalg.norm(X_clean[i]-centroids[class_map[Y_clean[i]]])

    for j in range(len(poison_dataset)):
        dis_poison_to_centro[j] = np.linalg.norm(X_poison[j] - centroids[class_map[Y_poison[j]]])

    # the histogram of the data
    min = np.min([np.min(dis_clean_to_centro),np.min(dis_poison_to_centro)])
    max = np.max([np.max(dis_clean_to_centro),np.max(dis_poison_to_centro)])

    bins = np.linspace(min, max, 100)

    print(dis_clean_to_centro)
    plt.hist(dis_clean_to_centro, bins, alpha=0.5, label='clean data')
    plt.hist(dis_poison_to_centro, bins, alpha=0.5, label='poisoning data')
    plt.legend(loc='upper right')
    plt.title('The distance to the centroid of the clean data')
    plt.show()

