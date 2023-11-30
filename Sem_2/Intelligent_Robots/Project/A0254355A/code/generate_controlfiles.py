# from train import launch_env, teacher
from learner import NeuralNetworkPolicyCNNIntention
from model import ResnetCNNIntention, ResnetCNNIntentionPadded
import argparse
import os
import json
import numpy as np
import sys
import cv2
import time
import webcolors
from highlevelplanner.base_hlp import HighLevelPlanner
from intelligent_robots_project import LaneFollower
from utils.vizmap import viz_map
from pprint import pprint

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'gym-duckietown')

from gym_duckietown.envs import DuckietownEnv
from termcolor import colored
from lanedetector import get_white_lanes, get_yellow_lanes,get_lines
import random

import torch
def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

TESTCASE_PATH = "../../testcases"
DEFAULT_DISCRETE_INTENTIONS = ["forward", "right", "left"]


def intention_to_idx(intention): 
    return DEFAULT_DISCRETE_INTENTIONS.index(intention)

def idx_to_intention(idx): 
    return DEFAULT_DISCRETE_INTENTIONS[idx]

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-i', default=10, type=int)
    parser.add_argument('--horizon', '-r', default=64, type=int)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)
    parser.add_argument('--model-path-forward', '-mpf', default="model/model_lanefollower.pt", type=str)
    parser.add_argument('--model-path-aligner', '-mpa', default="model/model_aligner.pt", type=str)
    parser.add_argument('--map-name', '-m', default="map5_0", type=str)

    return parser

def load_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key in testcases.keys():
            testcases[key]["map_name"] = key
        return testcases
    
def get_milestone_paths(milestone):
    path_name = f"{milestone['map_name']}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"

    subgoals = {}
    with open(os.path.join(TESTCASE_PATH, "milestone2_paths", path_name)) as fp:
        subgoals_strings = fp.read().split("\n")

        intention_list = []
        state_list = []
        for s in subgoals_strings:
            if len(s) > 0:
                intention = s.split(" ")[2].strip(" ")
                intention_list.append(intention)

                state = (int(s.split(',')[0][1:]), int(s.split(',')[1].split(')')[0].strip(" ")))
                state_list.append(state)

        for idx, s in enumerate(state_list[:-1]):
            subgoals[s] = intention_list[idx+1] 

    return subgoals


if __name__ == '__main__':
    seed(0)
    parser = process_args()
    input_shape = (120,160)
    max_velocity = 0.6 # <- This is not used.
    max_steering = 1

    config = parser.parse_args()

    testcases = load_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))
    milestone = testcases[config.map_name]
    milestone_path = get_milestone_paths(milestone)
    
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=50000,
        map_name=config.map_name,
        seed=int(milestone['seed'][0]),
        user_tile_start=milestone['start'],
        goal_tile=milestone['goal'],
        randomize_maps_on_reset=False
    )


    if not(os.path.isfile(config.model_path_forward)):
        raise Exception('Model File not found')
    else:
        print(f"loading control policy from {config.model_path_forward}")

    # model = ResnetCNNIntentionPadded(num_outputs=config.num_outputs, max_velocity=max_velocity)
    model_forward = ResnetCNNIntention(num_outputs=config.num_outputs, max_velocity=max_velocity)
    policy_forward = NeuralNetworkPolicyCNNIntention(
        model=model_forward,
        optimizer= None,
        scheduler=None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        max_steering = max_steering,
        model_path = config.model_path_forward
    )

    model_aligner = ResnetCNNIntentionPadded(num_outputs=config.num_outputs, max_velocity=max_velocity)
    imitation_aligner = NeuralNetworkPolicyCNNIntention(
        model=model_aligner,
        optimizer= None,
        scheduler=None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        max_steering = max_steering,
        model_path = config.model_path_aligner
    )
  
    control_file = []
    reward_file = []

    env.seed(int(milestone['seed'][0]))
    obs = env.reset()
    prev_intention = None
    intention = 'forward'
    info = {"curr_pos": None}
    done = False
    t = 0
    
    action = np.array([0,0])
    obs, reward, done, info = env.step(action)
    map_img, goal, start_pos = env.get_task_info()
    lanefollower_aligner = LaneFollower(milestone_path, map_img, goal, visualize=False)

    # First Align bot into lane using LAC
    print(colored("ALIGNING INTO LANE", "red"))

    action = np.array([0,0])
    obs, reward, done, new_info = env.step(action)
    while True:
        if t < 0.05:
            action = lanefollower_aligner(obs, new_info, action)
        else:
            action = imitation_aligner.predict(obs, [intention_to_idx(intention), t]).cpu().numpy()

        obs, reward, done, new_info = env.step(action)
        if reward < -1.2:
            time.sleep(0.1)
        print(f"Intention {colored(intention, 'yellow')} t={t:.3f} |\t Action {action[0]:.3f},{action[1]:.3f} |\t {colored(new_info['curr_pos'], 'green')} |\t {colored(reward, 'blue')}")
        control_file.append(action)
        reward_file.append(reward)
        t = np.clip(t+ 1/100, 0, 1)
        if info['curr_pos'] != new_info['curr_pos']:
            t=0
            break
        cv2.imshow("Observations", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


    alignment_timesteps_required = len(control_file)
    alignment_reward = np.sum(reward_file)

    # GENERATE HLP ON THE FLY

    def get_hlp(env, info, new_info, milestone):
        print(colored("GENERATING HIGH LEVEL PATH", "red"))
        print(colored(f"First tile {info['curr_pos']} Second tile {new_info['curr_pos']}", "red"))

        hlp = HighLevelPlanner(env, info['curr_pos'], new_info['curr_pos'], milestone['goal'])
        high_level_path = hlp.perform_a_star()

        milestone_path = {}
        for idx, path in enumerate(high_level_path):
            if idx > 0:
                tile_idx = high_level_path[idx-1][0]
                intention_idx = high_level_path[idx][1]
                milestone_path[tile_idx] =  intention_idx
        pprint(milestone_path)
        return milestone_path
    
    milestone_path = get_hlp(env, info, new_info, milestone)
    lanefollower_aligner = LaneFollower(milestone_path, map_img, goal, visualize=False)


    # After bot is aligned use LFC and ITC
    print(colored("FOLLOWING LANE TO GOAL", "magenta"))
    t = 0
    info = new_info
    tile_count = 1 # The nth tile the agent has visited
    while not done:
        intention = milestone_path[new_info["curr_pos"]]
        # Supply Generic Adaptive controls
        if tile_count == 1:
            action = imitation_aligner.predict(obs, [intention_to_idx(intention), t]).cpu().numpy()
        else:
            action = policy_forward.predict(obs, [intention_to_idx(intention), t]).cpu().numpy()

        if intention == "forward":
            if prev_intention == "forward" and tile_count != 1:
                # Already in lane move fast
                action += np.array([+0.2, 0])
            elif t>=0.15 and tile_count != 1:
                # Give time to stabilize before increasing speed
                action += np.array([+0.2, 0])

        if intention == "right" and t <= 0.2:
            action += np.array([+0.1, -1.5+t])
        elif intention == "left" and t<0.1:
            action = np.array([0.7, action[1]+0.3])
        elif intention == "left" and t>=0.1 and t <= 0.3:
            action += np.array([+0.1, +1.5 - 2*(t-0.1)])

        t = np.clip(t+ 1/100, 0, 1)

        obs, reward, done, new_info = env.step(action)
        if reward < -1.2:
            time.sleep(0.1)
        control_file.append(action)
        reward_file.append(reward)
        print(f"Tile {tile_count} Intention {colored(intention, 'yellow')} t={t:.3f} |\t Action {action[0]:.3f},{action[1]:.3f} |\t {colored(new_info['curr_pos'], 'green')} |\t {colored(reward, 'blue')}")

        if list(new_info['curr_pos']) == milestone['goal']:
            print("Goal Reached")
            ## Add localization
            done = True
            break
        else:
            done = False

        if info["curr_pos"] != new_info["curr_pos"]:
            if new_info["curr_pos"] not in milestone_path:
                milestone_path = get_hlp(env, info, new_info, milestone) 
            info = new_info
            prev_intention = intention
            t = 0
            tile_count += 1
        
        
        viz_map(env, info, new_info)

        cv2.putText(obs, text=f"Intention {intention} T={t:0.3f}", 
                    org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=webcolors.name_to_rgb("white"), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(obs, text=f"Action {action}", 
                    org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=webcolors.name_to_rgb("white"), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("Observations", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        # env.render()
        
    # Save Controls
    control_file = np.array(control_file)
    control_path_name = f"../control_files_2/{config.map_name}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"
    np.savetxt(control_path_name, control_file, fmt="%1.10f", delimiter=",")
   
    reward_file = np.array(reward_file)
    reward_path_name = f"../reward_files_2/{config.map_name}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"
    reward_plot_name = f"../reward_files_2/{config.map_name}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.png"
    np.savetxt(reward_path_name, reward_file, fmt="%1.10f", delimiter=",")

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(np.arange(len(reward_file)), reward_file)
    plt.axhline(0, color='black')
    plt.axvline(alignment_timesteps_required, color='red')
    plt.title(f'Total Reward {np.sum(reward_file):0.3f}. Avg Reward {np.sum(reward_file)/len(reward_file):0.3f} \n \
              Alignment Reward{alignment_reward:0.3f} Path Reward {(np.sum(reward_file)-alignment_reward):0.3f}')
    plt.savefig(reward_plot_name)
    print("Done!")
    
    env.close()
