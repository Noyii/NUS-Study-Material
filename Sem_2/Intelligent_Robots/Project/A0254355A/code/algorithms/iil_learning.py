import random
import numpy as np
import torch
from termcolor import colored
class InteractiveImitationLearning:
    """
    A class used to contain main imitation learning algorithm
    ...
    Methods
    -------
    train(samples, debug)
        start training imitation learning
    """
    def __init__(self, env, teacher, learner, horizon, intention_horizon, intention_sampling_mode, episodes, trajectory_per_episode, test=False):
        """
        Parameters
        ----------
        env : 
            duckietown environment
        teacher : 
            expert used to train imitation learning
        learner : 
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """

        self.environment = env
        self.teacher = teacher
        self.learner = learner
        self.test = test

        # from IIL
        self._horizon = horizon
        self._intention_horizon = intention_horizon
        self._intention_sampling_mode = intention_sampling_mode 
        self._episodes = episodes
        self._trajectory_per_episode = trajectory_per_episode

        # data
        self._observations = []
        self._intentions = []
        self._expert_actions = []

        # statistics
        self.learner_action = None
        self.learner_uncertainty = None

        self.teacher_action = None
        self.active_policy = True  # if teacher is active

        # internal count
        self._current_horizon = 0
        self._episode = 0

        # event listeners
        self._episode_done_listeners = []
        self._found_obstacle = False
        # steering angle gain
        self.gain = 1 # TODO: What is steerting angle gain?

    def train(self, debug=False):
        """
        Parameters
        ----------
        teacher : 
            expert used to train imitation learning
        learner : 
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """
        self._debug = debug
        for episode in range(self._episodes):
            self._episode = episode
            # self._sampling()
            for _ in range(self._trajectory_per_episode):
                self._sampling()
                self.environment.reset()
                # self.environment.reset(random=False) if episode == 0 else self.environment.reset(random=True)
            self._optimize()  # episodic learning
            self._on_episode_done()

    def _sample_intention(self):
        if self._intention_sampling_mode == 'random':
            current_tile_pos = self.environment.get_grid_coords(self.environment.cur_pos)
            current_tile = self.environment._get_tile(*current_tile_pos)
            if '4way' in current_tile['kind']:
                return np.array([np.random.choice([0, 1, 2]), 0.0], dtype=np.float32)
            elif 'straight' in current_tile['kind']:
                return np.array([0, 0.0], dtype=np.float32)
            elif 'curve' in current_tile['kind']:
                return np.array([0, 0.0], dtype=np.float32)
            else:
                return np.array([0, 0.0], dtype=np.float32)
                # return np.array([np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2]), 0.0], dtype=np.float32)

                
        elif self._intention_sampling_mode == 'forward':
            return np.array([0, 0.0], dtype=np.float32)
        elif self._intention_sampling_mode == 'right':
            return np.array([1, 0.0], dtype=np.float32)
            # return np.array([np.random.choice([0, 1], p=[0.3, 0.7]), 0.0], dtype=np.float32)
        elif self._intention_sampling_mode == 'left':
            return np.array([2, 0.0], dtype=np.float32)
            # return np.array([np.random.choice([0, 2], p=[0.3, 0.7]), 0.0], dtype=np.float32)

    def _sampling(self):
        observation = self.environment.render_obs()
        info = {"curr_pos": None}
        intention = self._sample_intention()

        for horizon in range(self._horizon):
            self._current_horizon = horizon
            action = self._act(observation, intention)
            try:
                # Update obs and intention
                next_observation, reward, done, new_info = self.environment.step([action[0], action[1]*self.gain])
                intention[1] = np.clip(intention[1] + (1/self._intention_horizon), 0, 1)
                
                print(f"Intention {colored(intention[0], 'yellow')} t={intention[1]:.3f} |\t Action {action[0]:.3f},{action[1]:.3f} |\t {colored(new_info['curr_pos'], 'green')}")

                if info["curr_pos"] != new_info["curr_pos"]:
                    info = new_info
                    intention = self._sample_intention()
            except Exception as e:
                print(e)
            if self._debug:
                from utils.vizmap import viz_map, viz_obs
                viz_map(self.environment, info, new_info)
                viz_obs(next_observation, intention, action)
                # self.environment.render()
            observation = next_observation

    # execute current control policy
    def _act(self, observation, intention):
        if self._episode <= 1:  # initial policy equals expert's
            control_policy = self.teacher
        else:
            control_policy = self._mix()

        control_action = control_policy.predict(observation, intention)

        self._query_expert(control_policy, control_action, observation, intention)

        self.active_policy = control_policy == self.teacher
        if self.test:
            return self.learner_action

        return control_action

    def _query_expert(self, control_policy, control_action, observation, intention):
        if control_policy == self.learner:
            self.learner_action = control_action
        else:
            self.learner_action = self.learner.predict(observation, intention)

        if control_policy == self.teacher:
            self.teacher_action = control_action
        else:
            self.teacher_action = self.teacher.predict(observation, intention)

        if self.teacher_action is not None:
            self._aggregate(observation, intention, self.teacher_action)

        if self.teacher_action[0] < 0.1:
            self._found_obstacle = True
        else:
            self._found_obstacle = False

    def _mix(self):
        raise NotImplementedError()

    def _aggregate(self, observation, intention, action):
        if not(self.test):
            self._observations.append(observation)
            self._intentions.append(intention)
            self._expert_actions.append(action)

    def _optimize(self):
        if not(self.test):
            self.learner.optimize(
                self._observations, self._intentions, self._expert_actions, self._episode)
            print('saving model')
            self.learner.save()

    # TRAINING EVENTS

    # # triggered after an episode of learning is done
    # def on_episode_done(self, listener):
    #     self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        # for listener in self._episode_done_listeners:
        #     listener.episode_done(self._episode)
        self.environment.reset()
