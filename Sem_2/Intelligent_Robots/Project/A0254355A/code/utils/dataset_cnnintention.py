import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, List


class MemoryMapDatasetCNNIntention(Dataset):
    """Dataset to store multiple arrays on disk to avoid saturating the RAM"""
    def __init__(self, size: int, img_size: tuple, intention_size: tuple, target_size: tuple, path: str):
        """
        Parameters
        ----------
        size : int
            Number of arrays to store, will be the first dimension of the resulting tensor dataset.
        img_size : tuple
            Size of images,  (CxHxW) for images 
        intention_size : tuple
            Size of intentions, 2D/1D for features.
        target_size : tuple
            Size of the target, for our case it is a 1D array having, angular and linear speed.
        path : str
            Path where the file will be saved.
        """
        self.size = size
        self.img_size = img_size
        self.intention_size = intention_size
        self.target_size = target_size

        # Path for each array
        self.path = path
        self.image_path = os.path.join(path, 'image.dat')
        self.intention_path = os.path.join(path, 'intention.dat')
        self.target_path = os.path.join(path, 'target.dat')

        # Create arrays
        self.images = np.memmap(self.image_path, dtype='float32', mode='w+', shape=(self.size, *self.img_size))
        self.intentions = np.memmap(self.intention_path, dtype='float32', mode='w+', shape=(self.size, *self.intention_size))
        self.target = np.memmap(self.target_path, dtype='float32', mode='w+', shape=(self.size, *self.target_size))

        # Initialize number of saved records to zero
        self.length = 0

        # keep track of real length in case of bypassing size value
        self.real_length = 0

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get one pair of training examples from the dataset.

        Parameters
        ----------
        item : int
            Index on the first dimension of the dataset.

        Returns
        -------
        sample, target : tuple
            Training sample consisting of data, label of data_size and target_size, respectively.
        """
        images = torch.tensor(self.images[item, ...])
        intentions = torch.tensor(self.intentions[item, ...])
        target = torch.tensor(self.target[item, ...])

        return images, intentions, target

    def __len__(self) -> int:
        """Get size (number of saved examples) of the dataset.

        Returns
        -------
        length : int
            Occupied length of the dataset. Note that it returns the number of saved examples rather than the maximum
            size used in the initialization.
        """
        return self.length

    def extend(self, images: List[np.ndarray], intentions: List[np.ndarray], actions: List[np.ndarray]):
        """Saves observations to the dataset. Iterates through the lists containing matching pairs of observations and
        actions. After saving each sample the dataset size is readjusted. If the dataset exceeds its maximum size
        it will start overwriting the firs experiences.

        Parameters
        ----------
        images : List
            List containing np.ndarray images of size img_size.
        actions
            List containing np.ndarray actions of size target_size.
        """
        for index, (image, intention, action) in enumerate(zip(images, intentions, actions)):
            current_data_indx = self.real_length + index
            if self.real_length + index >= self.size:
                # it will be a circular by getting rid of old experiments
                current_data_indx %= self.size 
            self.images[current_data_indx, ...] = image.astype(np.float32)
            self.intentions[current_data_indx, ...] = intention.astype(np.float32)
            self.target[current_data_indx, ...] = action.astype(np.float32)
        if self.real_length >= self.size:    
            self.length = self.size - 1
        else:
            self.length += len(images)
        self.real_length += len(images)

    def save(self):
        """In case of wanting to save the dataset this method should be implemented by flushing anc closing the memory
        map. Note that the files (depending on the size parameter) may occupy considerable amount of memory.
        """
        # TODO
        pass
