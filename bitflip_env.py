import numpy as np
import copy

class BitFlippingEnv():
    def __init__(self, bit_size):
        self.bit_size = bit_size

    def reset(self):
        self.state = np.random.randint(2, size=(self.bit_size))
        self.goal = np.random.randint(2, size=(self.bit_size))
        return copy.deepcopy(self.state), copy.deepcopy(self.goal)

    def step(self, action):

        if self.state[action] == 1:
            self.state[action] = 0

        elif self.state[action] == 0:
            self.state[action] = 1

        done = np.array_equal(self.state, self.goal)

        if done: reward = 0
        else: reward = -1

        return copy.deepcopy(self.state), reward, done

    def render(self):
        print('state: ', self.state)
        print('goal: ', self.goal)