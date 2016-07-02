import numpy as np
from sklearn.datasets import fetch_mldata

import gym
from gym import spaces


class MNISTEnv(gym.Env):
    """
    The standard MNIST classification task framed as reinforcement learning task

    The MNIST dataset consists of 70000 images.
    Each image consists of 28x28 pixels and describes one digit.
    Only the 60000k images of the training set are used in the step function.
    """

    def __init__(self):
        mnist = fetch_mldata('MNIST original')
        # Rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        self.test_size = 60000
        self.X_train, self.X_test = X[:self.test_size], X[self.test_size:]
        self.y_train, self.y_test = y[:self.test_size].astype(int), y[self.test_size:].astype(int)
        self.action_space = spaces.Discrete(len(np.unique(self.y_train)))
        self.observation_space = spaces.Box(0, 1, self.X_train.shape[1])

    def _reset(self):
        self.current_step = 0
        self.current_indices = np.random.choice(self.test_size, 1000, replace=False)
        
        return self._get_current(self.X_train)

    def _step(self, action):
        reward = 1 if self._get_current(self.y_train) == action else 0
        self.current_step += 1

        if self.current_step >= self.spec.timestep_limit:
            done = True
            ob = self._get(self.X_train, -1)
            info = {"number": self._get(self.y_train, -1)}
        else:
            done = False
            ob = self._get_current(self.X_train)
            info = {"number": self._get_current(self.y_train)}
        return ob, reward, done, info

    def _get(self, array, index):
        return array[self.current_indices[index]]

    def _get_current(self, array):
        return self._get(array, self.current_step)

    def reset_test(self):
        self.current_test_step = 0
        self.current_correct = 0
        return self.X_test[self.current_test_step]

    def step_test(self, action):
        reward = 1 if self.y_test[self.current_test_step] == action else 0
        self.current_correct += reward
        self.current_test_step += 1
        acc = self.current_correct / float(self.current_test_step)
        

        if self.current_test_step >= len(self.y_test):
            done = True
            ob = self.X_test[-1]
            info = {"number": self.y_test[-1], "acc": acc}
        else:
            done = False
            ob = self.X_test[self.current_test_step]
            info = {"number": self.y_test[self.current_test_step], "acc": acc}
        return ob, reward, done, info
