import pygame

import numpy as np
import cv2

RED = (255,0,0)
GREEN= (0,255,0)
BLUE = (0,0,255)
DARKBLUE = (0,0,128)
WHITE = (255,255,255)
BLACK = (0,0,0)
PINK = (255,200,200)


class Visualizer(object):

    def __init__(self, block_size, height, width, color_mapping):
        '''
        :param block_size: size of a block in pixels.
        :param height: height of the screen in blocks
        :param width: width of the screen in blocks
        :param color_mapping: a dictionary mapping integers to color tuples.
        '''
        self.block_size = block_size
        self.h = height
        self.w = width
        self.color_mapping = color_mapping
        self.screen = pygame.Surface((self.w * block_size, self.h * block_size))

    def make_screen(self, object_array):
        '''
        :param object_array: 2d array of integers representing object types
        :return:
        '''
        for i in range(self.h):
            for j in range(self.w):
                color = self.color_mapping[object_array[i][j]]
                pygame.draw.rect(self.screen, color, (self.block_size*j, self.block_size*i, self.block_size, self.block_size), 0)
        pixels = np.transpose(pygame.surfarray.array3d(self.screen), [1, 0, 2])[:, :, [2, 0, 1]]
        return pixels

v = Visualizer(4, 4, 4, {0: WHITE, 1: BLUE})
screen = v.make_screen([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0]])

cv2.imwrite('./test.png', cv2.resize(screen, (400, 400), interpolation=cv2.INTER_NEAREST))
