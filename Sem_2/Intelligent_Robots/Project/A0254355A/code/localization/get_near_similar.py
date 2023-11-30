"""
This files tests all milestone and calculates the total reward
"""

import argparse, json, os
import numpy as np
import sys
import pyglet
import cv2
import csv
sys.path.insert(1, '../gym-duckietown')

from termcolor import colored
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from utils import *
from os import listdir
from os.path import isfile, join


TESTCASE_PATH = "../../../testcases"
CONTROL_PATH = "../../controls_final"
DATA_PATH = "goal_tile_images/duck"

TANK_LEFT = np.array([0, +1.5])
TANK_RIGHT = np.array([0, -1.5])
FORWARD = np.array([0.5, 0])

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', '-m', default="map1_0", type=str)

    return parser

def load_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key in testcases.keys():
            testcases[key]["map_name"] = key
        return testcases

def update(dt):
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.0, +1.5])
    if key_handler[key.RIGHT]:
        action = np.array([0.0, -1.5])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost when pressing shift
    if key_handler[key.LSHIFT]:
        action *= 3

    _, reward, _, info = env.step(action)
    print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    env.render()


if __name__ == '__main__':
    parser = process_args()
    config = parser.parse_args()
    window = None
    
    testcases = load_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))
    milestone = testcases[config.map_name]
    reward = 0

    env = DuckietownEnv(
        domain_rand=False,
        max_steps=50000,
        map_name=milestone['map_name'],
        seed=int(milestone['seed'][0]),
        user_tile_start=milestone['start'],
        goal_tile=milestone['goal'],
        randomize_maps_on_reset=False
    )
    env.seed(int(milestone['seed'][0]))
    env.reset()

    if window == None:
        window = env.window
    else:
        env.window = window

    path = f"{milestone['map_name']}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"
    control_path = os.path.join(CONTROL_PATH, path)
    actions = np.loadtxt(control_path, delimiter=',')

    total_timesteps = len(actions)
    total_reward = 0 
    for speed, steering in actions:
        obs, reward, done, info = env.step([speed, steering])
        total_reward += reward
        env.render()

    print(f"Milestone {milestone['map_name']}", \
            "Total Reward", colored(total_reward, 'green'), \
            "Total Timesteps",  colored(total_timesteps, 'green'))
    
    print(colored("Let's find the duckie now :)", 'red'))
    control_file = []
    for _ in range(5):
        obs, reward, done, new_info = env.step(FORWARD)
        control_file.append(FORWARD)
        env.render()

    t = 0
    found = False
    map_images = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and milestone['map_name'] in f]

    for _ in range(105):
        obs, reward, _, new_info = env.step(TANK_RIGHT)
        control_file.append(TANK_RIGHT)
        env.render()

        img = preprocess_obs(obs)
        scores = []
        for m in map_images:
            goal_image = cv2.imread(join(DATA_PATH, m))
            score = check_image_similarity(img, goal_image)
            scores.append(score)

        scores = np.array(scores)
        print(scores)
        if np.max(scores) > 0.90:
            print(np.max(scores), map_images[np.argmax(score)])
            print(colored("Found you duck!", 'yellow'))
            break
    
    # print(new_info)
    while reward > -5 and list(new_info['curr_pos']) == milestone['goal']:
        print(colored("Moving forward", 'green'))
        action = FORWARD
        obs, reward, done, new_info = env.step(action)
        control_file.append(action)
        env.render()
    
    control_file = control_file[:-1]

    # key_handler = key.KeyStateHandler()
    # env.unwrapped.window.push_handlers(key_handler)
    # pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    # pyglet.app.run()

    # print(new_info)

    # Save Controls
    control_file = np.array(control_file)
    f = open(control_path,'a')    
    np.savetxt(f, control_file, fmt="%1.10f", delimiter=",")
    f.close()
    
    env.close()

