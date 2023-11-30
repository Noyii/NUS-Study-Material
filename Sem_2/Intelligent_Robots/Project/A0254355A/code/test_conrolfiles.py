"""
This files tests all milestone and calculates the total reward
"""

import argparse, json, os
import numpy as np
from termcolor import colored
from gym_duckietown.envs import DuckietownEnv

TESTCASE_PATH = "../../testcases"
CONTROL_PATH = "../control_files_2"

def process_args():
    parser = argparse.ArgumentParser()
    return parser

def laod_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key, values in testcases.items():
            testcases[key]["map_name"] = key
        return testcases


if __name__ == '__main__':
    parser = process_args()
    config = parser.parse_args()

    alphas = [2,3,4,5,6]
    V = 0

    window = None

    testcases = laod_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))

    for idx, (_, milestone) in enumerate(testcases.items()):
        a = alphas[idx // 5]
        try:
            # Metrics
            reward = 0
            alignment_reward = 0

            env = DuckietownEnv(
                domain_rand=False,
                max_steps=50000,
                map_name=milestone['map_name'],
                seed=int(milestone['seed'][0]),
                user_tile_start=milestone['start'],
                goal_tile=milestone['goal'],
                randomize_maps_on_reset=False
            )


            if window == None:
                window = env.window
            else:
                env.window = window

            control_path = os.path.join(CONTROL_PATH,  f"{milestone['map_name']}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt")
            actions = np.loadtxt(control_path, delimiter=',')


            total_timesteps = len(actions)
            total_reward = 0 
            for speed, steering in actions:
                obs, reward, done, info = env.step([speed, steering])

                # print(f"current pose = {info['curr_pos']}, action {speed, steering} step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
                total_reward += reward
                if info['curr_pos'][0] == milestone['start'][0]  and info['curr_pos'][1] == milestone['start'][1]:
                    alignment_reward += reward
                    
                # env.render()
            
            env.close()

            abs_avg_reward = np.clip(total_reward/total_timesteps, 0, 1000)
            print(f"Milestone {milestone['map_name']}", \
                    "Align Reward", colored(f"{alignment_reward:0.3f}", 'green'), \
                    "\tPath Reward", colored(f"{(total_reward-alignment_reward):0.3f}", 'green'), \
                    "\tTotal Reward", colored(f"{total_reward:0.3f}", 'green'), \
                    "\tTotal Timesteps",  colored(f"{total_timesteps:0.3f}", 'green'), \
                    "\tAvg Reward",  colored(f"{total_reward/total_timesteps:0.3f}", 'yellow'), \
                    "\tAbs Avg Reward",  colored(f"{abs_avg_reward:0.3f}", 'red' if abs_avg_reward == 0 else ('green' if abs_avg_reward > 0.5 else 'yellow')))

            V += a * abs_avg_reward
        except Exception as e:
            print(colored(f"Milestone {milestone['map_name']} failed: {e}", 'red'))
    print(f"V Score {V/np.sum(alphas)}")