import numpy as np
from sklearn.datasets import make_blobs


import gym
from gym import spaces


class ContextualBanditEnv(gym.Env):
    """
    Contextual bandit environment

    There are 3 classes of 8 arms each.
    The agent observes a point depicted by 2 features.
    Depending on the class of the point a different set of probabilities for the 8 arms is used.
    """

    def __init__(self):
        self.n_samples = 3000
        self.n_features = 2
        self.center_box = (-10.0, 10.0)
        self.n_classes = 3
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(self.center_box[0], self.center_box[1], self.n_features)

    def _reset(self):
        self.current_step = 0
        
        # Generate points for observations
        X, y = make_blobs(self.n_samples, n_features=self.n_features, centers=self.n_classes, 
                            center_box=self.center_box, shuffle=True)
        self.observations = X
        self.class_of_observations = y
        
        # Generate probabilities
        self.probabilities = np.random.uniform(0, 1, (self.n_classes, self.action_space.n))
        
        # Return first observation
        return self.observations[self.current_step]

    def _step(self, action):
        ob_class = self.class_of_observations[self.current_step]
        reward = np.random.binomial(1, self.probabilities[ob_class, action])
        done = False

        self.current_step += 1
        ob = None
        if self.current_step >= self.n_samples:
            done = True
        else:
            ob = self.observations[self.current_step]
        return ob, reward, done, {"probabilities": self.probabilities}
