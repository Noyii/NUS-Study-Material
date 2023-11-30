# README

# Problem Setup
In this problem, you are required to solve the Elevator environment using any discrete RL algorithm. 
The Elevator environment for this problem has the same setting as in assignment 2, i.e., the same states, actions, and rewards.

# Instructions For Submission
To ensure uniformity and ease of evaluation, please follow the steps below for
your model submission

1. Model Definition
    - Use `agent.py` to store your model's definition.
    - Your model MUST subclass from nn.Module.
    - Your model should take in 11 inputs, which will be derived from the convert_state_to_list function.
    - Your model should return 6 values corresponding to action logits or probabilities.

```
import torch.nn as nn

# Define the Model here - all component models (in case of actor-critic or others) MUST subclass nn.Module
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Your model layers and initializations here

    def forward(self, x):
        # x will be a tensor with shape [batch_size, 11]
        # Your forward pass logic here
        # Ensure the output has shape [batch_size, 6]
        return output
```

2. Training Your Model
    - Make use of the provided template notebook to train your model (https://colab.research.google.com/drive/1W4WkoRJcbcj91Sl1xPTMSqlvOVkjDTgJ?usp=sharing).
    - Throughout the training process in the notebook, a `model.pt` file will be
      generated, which contains the saved weights of your model.
    
3. Submission
    - Once you've trained your model and are satisfied with its performance, upload the following to AiRENA:
    - Your filled-out `agent.py`.
    - The saved `model.pt` file.
    - Be aware that there are limited submissions possible per day per team member. Plan your
      submissions wisely!
    
Good luck!
