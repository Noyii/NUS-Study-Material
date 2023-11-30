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


TESTCASE_PATH = "../../../testcases"
CONTROL_PATH = "../../controls_final"
DATA_PATH = "goal_tile_images/noDuck/noDuck_"

TANK_LEFT = np.array([0, +1.5])
TANK_RIGHT = np.array([0, -1.5])
FORWARD = np.array([0.5, 0])


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', '-m', default="map4_0", type=str)

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
        action = np.array([0.0, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.0, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost when pressing shift
    if key_handler[key.LSHIFT]:
        action *= 3

    _, reward, _, info = env.step(action)
    print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    env.render()

def extract_color(img, color):
    brightness = 0
    contrast = 10
    brightness += int(round(255*(1-contrast)/2))
    img = cv2.addWeighted(img, contrast, img, 0, brightness)

    img = cv2.GaussianBlur(img, (5, 5), 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # set lower and upper color limits
    if color == 'blue':
        lower_val = (90, 50, 20)
        upper_val = (110, 100, 255) # building glass
    elif color == 'grey':
        lower_val = (0, 0, 20)
        upper_val = (180, 15, 255) # grey 
    else:
        print(colored("Colour not specified", "red"))
    
    mask = cv2.inRange(hsv, lower_val, upper_val)
    edges = cv2.Canny(mask, 100, 100)
    # apply mask to original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # detect contours in image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    areas = [0]

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        
        if len(approx) > 4:
            ar = cv2.contourArea(cnt)
            areas.append(ar)
            count+=1
            img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)

    sorted_areas = sorted(areas, reverse=True)
    # diff = sorted_areas[0] - sorted_areas[1]
    return count, sorted_areas[0]
    # return count, (sorted[0] - sorted[1])


def detect_building(img, object='block'):
    count, max_area = extract_color(img, 'grey')
    # count, max_area, diff_area = extract_color(img, 'blue')
    print(max_area)

    if max_area > 30000:
        return True

    # if object == 'glass':
    #     count = extract_color(img, 'grey')
    #     print(count)
    #     if count > 600:
    #         print("Found it!")
    #         return True
    # else:
    #     count = extract_color(img, 'grey')
    #     print(count)
    #     if count > 450:
    #         print("Found it!")
    #         return True
    
    return False


if __name__ == '__main__':
    window = None
    parser = process_args()
    config = parser.parse_args()
    testcases = load_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))
    milestone = testcases[config.map_name]
    
    # for _, milestone in testcases.items():
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
        # env.render()

    print(f"Milestone {milestone['map_name']}", \
            "Total Reward", colored(total_reward, 'green'), \
            "Total Timesteps",  colored(total_timesteps, 'green'))
    
    print(colored("Let's find the duckie now :)", 'yellow'))

    for _ in range(5):
        obs, reward, done, new_info = env.step(FORWARD)
        # env.render()
    
    count = 0
    for _ in range(105):
        obs, _, _, _ = env.step(TANK_RIGHT)
        # env.render()
        # print(count)

        count += 1
        img = preprocess_obs(obs)
        path_name = DATA_PATH + milestone['map_name'] + '_' + str(count) + '.png'
        cv2.imwrite(path_name, img)
    
    # env.close()
    print(milestone['map_name'] + " done!")

