import numpy as np
import visualizer
import cv2
from random import choice

IMAGE = 0
VECTOR = 1

class Walk1D(object):

    def __init__(self, length, prob_left, mode):
        assert 0 <= prob_left <= 1
        assert mode in [IMAGE, VECTOR]
        self.mode = mode
        self.n = length
        self.p = prob_left
        self.state = np.random.randint(0, length)
        self.visualizer = visualizer.Visualizer(4, 1, self.n,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE})

    def set_state(self, state):
        self.state = state

    def get_state(self):
        onehot = [0] * self.n
        onehot[self.state] = 1
        if self.mode == IMAGE:
            return self.visualizer.make_screen([onehot])
        else:
            return np.array(onehot)

    def update(self):
        delta = -1 if np.random.uniform(0, 1) < self.p else 1
        self.state = np.clip(self.state+delta, 0, self.n)
        return self.get_state()


class BitFlip1D(object):

    def __init__(self, length, mode):
        assert mode in [IMAGE, VECTOR]
        self.mode = mode
        self.n = length
        self.state = np.random.randint(0, 1, size=(1, self.n))
        self.visualizer = visualizer.Visualizer(4, 1, self.n,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE})

    def set_state(self, state):
        self.state = state

    def get_state(self):
        if self.mode == IMAGE:
            return self.visualizer.make_screen(self.state)
        else:
            return self.state.copy()[0, :]

    def update(self):
        bit_to_flip = np.random.randint(0, self.n)
        self.state[0, bit_to_flip] = 1 - self.state[0, bit_to_flip]
        return self.get_state()

class Walk2D(object):

    def __init__(self, width, height, prob_dirs, obstacle_pos_list, mode, should_wrap):
        self.UP, self.DOWN, self.LEFT, self.RIGHT = 0, 1, 2, 3
        assert sum(prob_dirs) == 1
        self.h = height
        self.w = width
        self.prob_dirs = prob_dirs
        self.action_delta_mapping = {
            self.UP: (0, -1),
            self.DOWN: (0, 1),
            self.LEFT: (-1, 0),
            self.RIGHT: (1, 0)
        }
        self.obstacle_pos_list = obstacle_pos_list
        non_obstacle_squares = [(x, y) for x in range(self.w) for y in range(self.h)
                                if (x, y) not in self.obstacle_pos_list]
        self.should_wrap = should_wrap
        self.mode = mode
        self.x_pos, self.y_pos = choice(non_obstacle_squares)
        self.visualizer = visualizer.Visualizer(4, height, width,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE,
                                                 2: visualizer.BLACK})


    def set_state(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        assert (x_pos, y_pos) not in self.obstacle_pos_list

    def get_state(self):
        array = np.zeros((self.h, self.w), dtype=np.uint8)
        array[self.y_pos, self.x_pos] = 1
        for obs_x, obs_y in self.obstacle_pos_list:
            array[obs_y, obs_x] = 2
        if self.mode == IMAGE:
            return self.visualizer.make_screen(array)
        else:
            return array

    def update(self):
        action = np.random.choice([self.UP, self.DOWN, self.LEFT, self.RIGHT], p=self.prob_dirs)
        (delta_x, delta_y) = self.action_delta_mapping[action]
        if self.should_wrap:
            new_x_pos = (self.x_pos+delta_x) % self.w
            new_y_pos = (self.y_pos+delta_y) % self.h
        else:
            new_x_pos = np.clip(self.x_pos+delta_x, 0, self.w)
            new_y_pos = np.clip(self.y_pos+delta_y, 0, self.h)
        if (new_x_pos, new_y_pos) not in self.obstacle_pos_list:
            self.x_pos = new_x_pos
            self.y_pos = new_y_pos
        return self.get_state()


# 1D random walk domain with length-5, 50 percent chance of moving left, represented as VECTOR
domain = Walk1D(5, 0.5, VECTOR)
# 1D random walk domain with length-5 50 percent change of moving right, represented as IMAGE
domain = Walk1D(5, 0.5, IMAGE)
# 1D random bit flip domain with length-5 represented as IMAGE
domain = BitFlip1D(5, IMAGE)
# 2D 5x5 random walk domain with norvig move-dynamics, no obstacles, a vector representation, and wrapping.
domain = Walk2D(5, 5, [0.8, 0.0, 0.1, 0.1], [], VECTOR, True)
# 2D 5x5 random walk domain with norvig move-dynamics, an obstacle in the center, a vector representation, and no wrapping.
domain = Walk2D(5, 5, [0.8, 0.0, 0.1, 0.1], [(2, 2)], VECTOR, False)
