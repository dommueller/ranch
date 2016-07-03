import numpy as np

import gym
from gym import spaces


class MultiArmedBanditRegretEnv(gym.Env):
    """
    Simple multi armed bandit environment

    There are 8 arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and gets a reward of 1 with probabily p_i.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Discrete(2)

    def _reset(self):
        self.steps = 0
        self.summed_reward = 0
        self.probabilities = np.random.uniform(0, 1, self.action_space.n)
        return 0

    def _step(self, action):
        self.steps += 1
        self.summed_reward += np.random.binomial(1, self.probabilities[action])
        if self.steps >= self.spec.timestep_limit:
            regret = self.steps * np.max(self.probabilities) - self.summed_reward
            return self.summed_reward, -regret, False, {"probabilities": self.probabilities}
        else:
            return self.summed_reward, 0, False, {"probabilities": self.probabilities}
