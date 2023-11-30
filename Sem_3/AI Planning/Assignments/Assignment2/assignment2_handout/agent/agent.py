
import numpy as np
# from elevator import ElevatorEnv
from pyRDDLGym.Policies.Agents import RandomAgent
from pyRDDLGym.Elevator import Elevator
from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import SampleSerializer
from PIL import Image


"""
Elevator enviroment for Value Iteration. Please read README careful to
understand the problem setup
"""
class ElevatorAgentEnv(AgentEnv):
    def __init__(self, port: int):
        self.base_env = Elevator()

        super().__init__(
            SampleSerializer(),
            self.base_env.action_space,
            self.base_env.observation_space,
            self.base_env.reward_range,
            uid=0,
            port=port,
            env=self.base_env,
        )  # uid can be any int for single-agent agent env

    def create_agent(self, **kwargs):
        agent = ValueIterationAgent(env=self.base_env, gamma=0.99, theta=0.000001, max_iterations=10000)
        agent.initialize()

        return agent

class ValueIterationAgent(object):
    def __init__(self, env=None, 
                       gamma=0.99,
                       theta = 0.00001,
                       max_iterations=10000):

        self.env = env
                           
        # Set of discrete actions for evaluator environment, shape - (|A|)
        self.disc_actions = env.disc_actions

        # Set of discrete states for evaluator environment, shape - (|S|)
        self.disc_states = env.disc_states

        # Set of probabilities for transition function for each action from every states, dicitonary of dist[s] = [s', prob, done, info]
        self.Prob = env.Prob

        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        # self.value_policy, self.policy_function = None, None

    def initialize(self):
        self.value_policy, self.policy_function = self.solve_value_iteration()


    def step(self, state):
        action = self.policy_function[int(state)]
        return action


    def solve_value_iteration(self):
        '''
        FILL_ME: implement value iteration here
        Tips: 
            - You can create as many new functions within this class
            - Use self.disc_actions and self.disc_states to access actions and states respectively
            - Use self.Prob to get transition probability between states for given action.

        NOTE: Do not modify other existing methods/classes. Please ask in the forums if you have questions/need clarification

        return:
            value_policy (shape - (|S|)): utility value for each state
            policy_function (shape - (|S|), dtype = int64): action policy per state
        '''

        # the following is random policy provided for testing purpose only
        # your proposed solution must be better than random

        value_policy = np.random.choice(len(self.env.disc_actions), size=len(self.env.disc_states))
        policy_function = np.random.choice(len(self.env.disc_actions), size=len(self.env.disc_states))

        return value_policy, policy_function


def main():

    # Regarding rendering: we save each step as png and convert to png under the hood. 
    # Set is_render=True to do so or your local testing
    # before submitting, always set is_render to False.
    is_render = False
    render_path = 'temp_vis'
    env = Elevator(is_render=is_render, render_path=render_path)

    agent_env = ElevatorAgentEnv(0)
    agent = agent_env.create_agent()
    state = env.reset()

    total_reward = 0
    for t in range(env.horizon):
        action = agent.step(state)
        next_state, reward, terminated, info = env.step(action)  # self.env.step(action)

        if is_render:
            env.render()
            
        total_reward += reward
        print()
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}')
        print(f'total_reward     = {total_reward}')
        state = next_state

    env.close()
    

    if is_render:
        env.save_render()
        img = Image.open(f'{render_path}/elevator.gif').convert('RGB')
        img.show()
    

if __name__ == "__main__":
    main()
