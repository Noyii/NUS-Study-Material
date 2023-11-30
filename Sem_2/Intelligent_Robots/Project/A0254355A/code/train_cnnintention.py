import math
from gym_duckietown.envs import DuckietownEnv
import argparse
import re
import numpy as np
from teacher import PurePursuitPolicy
from learner import NeuralNetworkPolicyCNNIntention
from model import ResnetCNNIntention, ResnetCNNIntentionPadded
from algorithms import DAgger
from utils.dataset_cnnintention import MemoryMapDatasetCNNIntention
from utils.multimap_env_imitation import MultiMapImiationEnv
import torch
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, resnet101, resnet152, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

def load_last_run(run_dir):
    # Setup logging
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    checkpoints = []
    checkpoints_ids = []
    for file in os.listdir(run_dir):
        checkpoints.append(file)
        checkpoints_ids.append(int(re.findall(r'\d+', file)[-1]))
    return np.amax(checkpoints_ids) if len(checkpoints_ids) else 0

def launch_multimap_env(randomize_maps_on_reset=False, domain_rand=False):
    environment = MultiMapImiationEnv(
        domain_rand=domain_rand,
        max_steps=math.inf,
        randomize_maps_on_reset=randomize_maps_on_reset
    )
    return environment

def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False):

    environment = DuckietownEnv(
        domain_rand=domain_rand,
        max_steps=math.inf,
        map_name=map_name,
        randomize_maps_on_reset=randomize_maps_on_reset
    )
    return environment
    
def teacher(env, max_velocity):
    return PurePursuitPolicy(
        env=env,
        ref_velocity=max_velocity
    )

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-i', default=10, type=int)
    parser.add_argument('--horizon', '-r', default=100, type=int)
    parser.add_argument('--intention-sampling-mode', '-ism', default='random', type=str)
    parser.add_argument('--learning-rate', '-l', default=2, type=int)
    parser.add_argument('--decay', '-d', default=4, type=int)
    parser.add_argument('--save-path', '-s', default='runs', type=str)
    parser.add_argument('--run-id', '-rid', default=None, type=str)
    parser.add_argument('--map-name', '-m', default="map5_0", type=str)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)
    return parser

if __name__ == '__main__':
    parser = process_args()
    input_shape = (120, 160)
    batch_size = 32
    epochs = 10
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # decays
    mixing_decays = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    
    
    # Max velocity
    max_velocity = 0.7
    max_steering = np.pi/2 # Steering angle to be predicted 

 
    config = parser.parse_args()
    # check for  storage path
    if not(os.path.isdir(config.save_path)):
        os.makedirs(config.save_path)

    # Create Run Path
    if not config.run_id:
        run_path = os.path.join(config.save_path, f"run_{load_last_run(config.save_path) + 1}") 
    else:
        run_path = os.path.join(config.save_path, config.run_id)
    if not(os.path.isdir(run_path)):
        os.makedirs(run_path)


    # launching environment
    # environment = launch_env(config.map_name)
    environment = launch_multimap_env()
    
    task_horizon = config.horizon
    task_episode = config.episode

    model = ResnetCNNIntentionPadded(
                num_outputs=config.num_outputs,
                max_velocity=max_velocity,
                max_steering=max_steering,
                backbone_net=resnet50,
                backbone_weights=ResNet50_Weights.IMAGENET1K_V2)
    
    policy_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[config.learning_rate])
    policy_scheduler = ReduceLROnPlateau(policy_optimizer, 'min', factor=0.1, patience=2)

    dataset = MemoryMapDatasetCNNIntention(25000, (3, input_shape[0], input_shape[1]), (2,), (2,), run_path)
    learner = NeuralNetworkPolicyCNNIntention(
        model=model,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        storage_location=run_path,
        batch_size=batch_size,
        epochs=epochs,
        input_shape=input_shape,
        max_velocity = max_velocity,
        dataset = dataset
    )

    algorithm = DAgger(env=environment,
                        teacher=teacher(environment, max_velocity),
                        learner=learner,
                        horizon = task_horizon,
                        intention_horizon = 100,
                        intention_sampling_mode = config.intention_sampling_mode,
                        episodes = task_episode,
                        alpha = mixing_decays[config.decay])
    
    algorithm.train(debug=True)  #DEBUG to show simulation

    environment.close()



