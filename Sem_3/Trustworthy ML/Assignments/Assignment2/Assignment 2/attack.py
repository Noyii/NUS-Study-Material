import copy

from utilities import *
import numpy as np
import torch


class Attack:
    def __init__(self, target_model, clean_dataset, test_dataset):
        self.target_model = target_model
        self.clean_dataset = clean_dataset
        self.test_dataset = test_dataset