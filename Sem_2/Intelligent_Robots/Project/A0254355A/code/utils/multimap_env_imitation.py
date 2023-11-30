# coding=utf-8
import os
import gym
import json
import random
from gym_duckietown.envs import DuckietownEnv

TEST_CASE_PATH = "/home/aishik/Codes/CS5477/IRProjectImpementation/testcases"

def laod_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key, values in testcases.items():
            testcases[key]["map_name"] = key
        return testcases

class MultiMapImiationEnv(gym.Env):
    """
    Environment which samples from multiple environments, for
    multi-taks learning
    """

    def __init__(self, **kwargs):
        self.env_list = []

        # self.config_list = []
        # self.config_list.extend(laod_milestone(os.path.join(TEST_CASE_PATH, "milestone1.json")).values())
        # self.config_list.extend(laod_milestone(os.path.join(TEST_CASE_PATH, "milestone2.json")).values())

        # unqiue_maps = list(set([config['map_name'] for config in self.config_list]))
        # unqiue_maps = ["map1_0", "map2_0", "map3_0", "map3_4", "map4_0", "map4_2", "map4_4", "map5_0", "map5_2", "map5_4"]
        self._seeds = {
            "map1_0": [{"seed": [1], "start": [0, 1], "goal": [5, 1]},
                        {"seed": [0], "start": [0, 1], "goal": [70, 1]},
                        {"seed": [2], "start": [2, 1], "goal": [21, 1]},
                        {"seed": [6], "start": [5, 1], "goal": [65, 1]},
                        {"seed": [5], "start": [50, 1], "goal": [90, 1]}],

            "map2_0": [{"seed": [1], "start": [7, 7], "goal": [1, 1]},
                        {"seed": [2], "start": [3, 6], "goal": [7, 1]},
                        {"seed": [5], "start": [1, 6], "goal": [3, 4]},
                        {"seed": [4], "start": [1, 2], "goal": [5, 4]},
                        {"seed": [4], "start": [7, 4], "goal": [4, 7]}],

            "map3_0": [{"seed": [1], "start": [5, 7], "goal": [2, 2]},
                        {"seed": [2], "start": [5, 11], "goal": [1, 7]},
                        {"seed": [3], "start": [10, 5], "goal": [7, 11]},
                        {"seed": [4], "start": [2, 4], "goal": [9, 1]},
                        {"seed": [12], "start": [5, 5], "goal": [10, 11]}],

            "map4_0": [{"seed": [4], "start": [10, 4], "goal": [3, 3]},
                        {"seed": [4], "start": [7, 7], "goal": [1, 12]},
                        {"seed": [4], "start": [4, 1], "goal": [11, 11]},
                        {"seed": [6], "start": [1, 8], "goal": [13, 8]},
                        {"seed": [8], "start": [5, 10], "goal": [11, 4]}],

            "map5_0": [{"seed": [0], "start": [10, 4], "goal": [2, 9]},
                        {"seed": [0], "start": [6, 8], "goal": [4, 13]},
                        {"seed": [2], "start": [10, 7], "goal": [10, 1]},
                        {"seed": [4], "start": [1, 6], "goal": [12, 15]},
                        {"seed": [5], "start": [3, 10], "goal": [15, 9]}],
        }
        self.unqiue_maps = ["map1_0", "map2_0", "map3_0", "map4_0", "map5_0",]

        self.window = None

        # Try loading each of the available map files
        for map_name in self.unqiue_maps:

            env = DuckietownEnv(
                map_name=map_name,
                **kwargs
            )

            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.reward_range = env.reward_range

            self.env_list.append(env)

        assert len(self.env_list) > 0

        self.cur_env_idx = 0
        self.cur_env_seed_idx = [0 for _ in range(len(self.env_list))]
        self.cur_reward_sum = 0
        self.cur_num_steps = 0

    # def seed(self, seed):
    #     for env in self.env_list:
    #         env.seed(seed)

    #     # Seed the random number generator
    #     self.np_random, _ = gym.utils.seeding.np_random(seed)

    #     return [seed]
    @property
    def map_name(self):
        return self.unqiue_maps[self.cur_env_idx]
    

    def _sample_seed(self, map_name):
        self.cur_env_seed_idx[self.cur_env_idx] = (self.cur_env_seed_idx[self.cur_env_idx] + 1) % 5
        return self._seeds[map_name][self.cur_env_seed_idx[self.cur_env_idx]]

    def reset(self):
        #self.cur_env_idx = self.np_random.randint(0, len(self.env_list))
        self.cur_env_idx = (self.cur_env_idx + 1) % len(self.env_list)

        env = self.env_list[self.cur_env_idx]
        seed = self._sample_seed(self.unqiue_maps[self.cur_env_idx])
        env.seed(int(seed['seed'][0]))
        env.user_tile_start = seed['start']
        env.goal_tile = seed['goal']
        return env.reset()

    def step(self, action):
        env = self.env_list[self.cur_env_idx]

        obs, reward, done, info = env.step(action)

        # Keep track of the total reward for this episode
        self.cur_reward_sum += reward
        self.cur_num_steps += 1

        # If the episode is done, sample a new environment
        if done:
            self.cur_reward_sum = 0
            self.cur_num_steps = 0

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        env = self.env_list[self.cur_env_idx]

        # Make all environments use the same rendering window
        if self.window is None:
            ret = env.render(mode, close)
            self.window = env.window
        else:
            env.window = self.window
            ret = env.render(mode, close)

        return ret

    def close(self):
        for env in self.env_list:
            env.close()

        self.cur_env_idx = 0
        self.env_names = None
        self.env_list = None

    def __getattr__(self, __name: str):
        if __name not in dir(self):
            env = self.env_list[self.cur_env_idx]
            return env.__getattribute__(__name)
        return __name
    
    

    @property
    def step_count(self):
        env = self.env_list[self.cur_env_idx]
        return env.step_count
