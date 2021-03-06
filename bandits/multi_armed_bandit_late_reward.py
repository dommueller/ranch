import numpy as np

import gym
from gym import spaces


class MultiArmedBanditLateRewardEnv(gym.Env):
    """
    Simple multi armed bandit environment

    There are n arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and wins with probabily p_i.
    At the end of the series the agent recieves the number of wins as reward.

    Actions:
    One action for each arm

    Observation:
    List that counts how often each arm was chosen
    List that counts how often each arm won
    """

    def __init__(self, n):
        self.action_space = spaces.Discrete(n)
        # First row tells how often each action was chosen
        # Second row tells how often each action has won
        shape = np.ones((2, n))
        self.observation_space = spaces.Box(shape * 0, shape * 2000)

    def _reset(self):
        self.steps = 0
        self.probabilities = np.random.uniform(0, 1, self.action_space.n)
        self.max_probability = np.max(self.probabilities)
        self.actions = [0 for _ in xrange(self.action_space.n)]
        self.results = [0 for _ in xrange(self.action_space.n)]
        return np.array([self.actions, self.results])

    def _step(self, action):
        self.steps += 1
        
        win = np.random.binomial(1, self.probabilities[action])
        self.actions[action] += 1
        self.results[action] += win
        
        ob = np.array([self.actions, self.results])
        if self.steps >= self.spec.timestep_limit:
            reward = sum(self.results)
            return ob, reward, True, {"probabilities": self.probabilities}
        else:
            return ob, 0, False, {"probabilities": self.probabilities}

    def _close(self):
        if hasattr(self, 'probabilities'):
            del self.probabilities
        if hasattr(self, 'actions'):
            del self.actions
        if hasattr(self, 'results'):
            del self.results

class FiveArmedBanditLateRewardEnv(MultiArmedBanditLateRewardEnv):
    """
    Simple multi armed bandit environment

    There are 5 arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and gets a reward of 1 with probabily p_i.
    """

    def __init__(self):
        super(FiveArmedBanditLateRewardEnv, self).__init__(5)

class EightArmedBanditLateRewardEnv(MultiArmedBanditLateRewardEnv):
    """
    Simple multi armed bandit environment

    There are 8 arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and gets a reward of 1 with probabily p_i.
    """

    def __init__(self):
        super(EightArmedBanditLateRewardEnv, self).__init__(8)

class TwentyArmedBanditLateRewardEnv(MultiArmedBanditLateRewardEnv):
    """
    Simple multi armed bandit environment

    There are 20 arms to choose from, each with an i.i.d. probability between 0 and 1.
    The agent chooses a arm to pull in each step and gets a reward of 1 with probabily p_i.
    """

    def __init__(self):
        super(TwentyArmedBanditLateRewardEnv, self).__init__(20)