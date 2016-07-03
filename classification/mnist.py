import numpy as np
import sys
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

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.test_size = 60000
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(0, 1, 28*28)

    def _reset(self):
        self.current_step = 0
        current_indices = np.random.choice(self.test_size, self.spec.timestep_limit, replace=False)
        mnist = fetch_mldata('MNIST original')
        # Rescale the data, use the traditional train/test split
        self.X_train, self.y_train = mnist.data[current_indices] / 255., mnist.target[current_indices].astype(int)
        return self._get_current(self.X_train)

    def _step(self, action):
        reward = 1 if self._get_current(self.y_train) == action else 0
        self.current_step += 1

        if self.current_step >= self.spec.timestep_limit:
            done = True
            ob = self.X_train[-1]
            info = {"number": self.y_train[-1]}
        else:
            done = False
            ob = self._get_current(self.X_train)
            info = {"number": self._get_current(self.y_train)}
        return ob, reward, done, info

    def _render(self, mode='human', close=False):
        outfile = sys.stdout
        outfile.write("\n".join("".join(["X" if x > 0. else "-" for x in line]) for line in self._get_current(self.X_train).reshape((28,28))) + "\n")
        outfile.write("Number: %d\n\n" % self._get_current(self.y_train))

    def _get_current(self, array):
        return array[self.current_step]

    def reset_test(self):
        mnist = fetch_mldata('MNIST original')
        # Rescale the data, use the traditional train/test split
        self.X_test, self.y_test = mnist.data[self.test_size:] / 255., mnist.target[self.test_size:].astype(int)
        return self._get_current(self.X_train)
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
