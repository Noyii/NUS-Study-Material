import argparse
import os
import json
import numpy as np
import sys
import cv2
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'gym-duckietown')

from gym_duckietown.envs import DuckietownEnv
from termcolor import colored
from highlevelplanner.base_hlp import HighLevelPlanner
from learner import NeuralNetworkPolicyCNNIntention
from model import ResnetCNNIntention

TESTCASE_PATH = "../../testcases"
DEFAULT_DISCRETE_INTENTIONS = ["forward", "right", "left"]


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', '-m', default="map5_0", type=str)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)
    parser.add_argument('--model-path-forward', '-mpf', default="model/model.pt", type=str)

    return parser


def load_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key in testcases.keys():
            testcases[key]["map_name"] = key
        return testcases
    

if __name__ == '__main__':
    parser = process_args()
    config = parser.parse_args()
    input_shape = (120,160)
    max_velocity = 0.7
    max_steering = 0.5
    
    testcases = load_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))
    milestone = testcases[config.map_name]
    init_pose = milestone['start']
    goal_pose = milestone['goal']
    seed = int(milestone['seed'][0])
    
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=50000,
        map_name=config.map_name,
        seed=seed,
        user_tile_start=init_pose,
        goal_tile=goal_pose,
        randomize_maps_on_reset=False
    )
    env.seed(seed)

    # env.render()
    # map_img, goal, start_pos = env.get_task_info()
    # cv2.imwrite('maps/' + env.map_name + ".png", map_img)
    
    model = ResnetCNNIntention(num_outputs=config.num_outputs, max_velocity=max_velocity)
    policy_forward = NeuralNetworkPolicyCNNIntention(
        model=model,
        optimizer= None,
        scheduler=None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        max_steering = max_steering,
        model_path = config.model_path_forward
    )

    hlp = HighLevelPlanner(env, init_pose, goal_pose, policy_forward)
    high_level_path = hlp.perform_a_star()
    
    # Save Path
    hlp_file = np.array(high_level_path)
    path_name = f"../../testcases/milestone2_paths/{config.map_name}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"
    np.savetxt(path_name, hlp_file, fmt='%s')
    print("Done!")
    
    env.close()
