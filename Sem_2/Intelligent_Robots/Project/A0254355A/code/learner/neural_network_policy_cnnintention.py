import numpy as np
from tqdm import tqdm
import cv2
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.transforms.functional import crop
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

class NeuralNetworkPolicyCNNIntention:
    """
    A wrapper to train neural network model
    ...
    Methods
    -------
    optimize(observations, expert_actions, episode)
        train the model on the newly collected data from the simulator
    
    predict(observation)
        takes images and predicts the action space using the model

    save
        save a model checkpoint to storage location
    """

    def __init__(self, model, optimizer, scheduler, storage_location, dataset, **kwargs):
        """
        Parameters
        ----------
        model : 
            input pytorch model that will be trained
        optimizer : 
            torch optimizer
        storage_location : string
            path of the model to be saved , the dataset and tensorboard logs
        dataset :
            object storing observation and labels from expert
        """
        self._train_iteration = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base parameters
        self.model = model.to(self._device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.storage_location = storage_location
        self.writer = SummaryWriter(self.storage_location)

        # Optional parameters
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.input_shape = kwargs.get('input_shape', (60, 80))
        self.max_velocity = kwargs.get('max_velocity', 0.7)

        self.episode = 0

        self.dataset = dataset
        # Load previous weights
        if 'model_path' in kwargs:
            self.model.load_state_dict(torch.load(kwargs.get('model_path'),map_location=self._device))
            print('Loaded ')

    def __del__(self):
        self.writer.close()

    def optimize(self, observations, intentions, expert_actions, episode):
        """
        Parameters
        ----------
        observations : 
            input images collected from the simulator
        intentions : 
            intentions for the input images collected from the simulator
        expert_actions : 
            list of actions each action [velocity, steering_angle] from expert
        episode : int
            current episode number
        """
        print('Starting episode #',str(episode))
        self.episode = episode
        self.model.episode = episode
        # Transform newly received data
        observations, expert_actions = self._transform(observations, expert_actions)

        # Retrieve data loader
        dataloader = self._get_dataloader(observations, intentions, expert_actions)
        # Train model
        for epoch in tqdm(range(1, self.epochs + 1)):
            running_loss = 0.0
            self.model.epoch = 0
            for i, data in enumerate(dataloader, 0):
                # Send data to device
                data = [variable.to(self._device) for variable in data]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.model.loss(*data)
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()

            # Logging
            self._logging(loss=running_loss, lr=self.optimizer.param_groups[0]['lr'], epoch=epoch)
        if self.scheduler != None:
            self.scheduler.step(running_loss)

        # Post training routine
        self.epochs += 2 # TODO: Test if we should increase epochs as training batch size grows
        self._on_optimization_end()

    def predict(self, observation, intentions):
        """
        Parameters
        ----------
        observations : 
            input image from the simulator
        Returns
        ----------
        prediction:
            action space of input observation
        """
        # Apply transformations to data
        observation, _ = self._transform([observation], [0])
        observation = torch.tensor(np.array(observation)).to(self._device)
        intentions = torch.tensor([intentions]).to(self._device)
        # Predict with model
        prediction = self.model.predict(observation, intentions)

        return prediction.cpu()

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.storage_location , 'model.pt'))

    def _transform(self, observations, expert_actions):
        # Resize images
        observations = [Image.fromarray(cv2.resize(observation, dsize=self.input_shape[::-1])) for observation in observations]
        # Transform to tensors
        compose_obs = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # using imagenet normalization values
        ])
        
        observations = [compose_obs(observation).numpy() for observation in observations]
        # observations = [observation[:, self.input_shape[0]//4:, :] for observation in observations]
        try:
            # scaling velocity to become in 0-1 range which is multiplied by max speed to get actual vel
            # also scaling steering angle to become in range -1 to 1 to make it easier to regress
            expert_actions = [np.array([expert_action[0],  expert_action[1]  / (np.pi/2) ]) for expert_action in expert_actions]
        except:
            pass
        expert_actions = [torch.tensor(expert_action).numpy() for expert_action in expert_actions]

        return observations, expert_actions

    def _get_dataloader(self, observations, intentions, expert_actions):
        # Include new experiences
        self.dataset.extend(observations, intentions, expert_actions)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _logging(self, **kwargs):
        epoch = kwargs.get('epoch')
        loss = kwargs.get('loss')
        lr = kwargs.get('lr')

        self.writer.add_scalars(f'CombinedLoss/train', {
            f"{self._train_iteration}": loss,
        }, epoch)

        self.writer.add_scalar('LR', lr, self._train_iteration)
        self.writer.add_scalar('Loss/train', loss, self._train_iteration * self.epochs + epoch)
        
        self.writer.add_scalars(f"Dataset Size", {
            "length": self.dataset.length,
            "real_length": self.dataset.real_length
        }, self._train_iteration)

        print(f"Epoch {epoch} Loss {loss} LR {lr}")

    def _on_optimization_end(self):
        self._train_iteration += 1
