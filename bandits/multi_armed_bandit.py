import numpy as np

import gym
from gym import spaces


class MultiArmedBanditEnv(gym.Env):
    """
    Simple multi armed bandit environment

    There are 8 arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and gets a reward of 1 with probabily p_i.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(1)

    def _reset(self):
        self.cum_reward = 0
        self.probabilities = np.random.uniform(0, 1, self.action_space.n)
        return 0

    def _step(self, action):
        reward = np.random.binomial(1, self.probabilities[action])
        return 0, reward, False, {"probabilities": self.probabilities}
