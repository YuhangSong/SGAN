from pygame.locals import *
import pygame
import numpy as np
import cv2
import sys


DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_NONE = range(5)

COLOR_RED = (255,0,0)
COLOR_GREEN= (0,255,0)
COLOR_BLUE = (0,0,255)
COLOR_DARKBLUE = (0,0,128)
COLOR_WHITE = (255,255,255)
COLOR_BLACK = (0,0,0)
COLOR_PINK = (255,200,200)

class GameLogic(object):

    def __init__(self, width=4, height=4, wrap=True):
        self.width = width
        self.height = height
        self.player_pos = (0, 0)
        self.wrap = wrap
        self.dir_mapping_y = {DIRECTION_DOWN: 1, DIRECTION_UP: -1}
        self.dir_mapping_x = {DIRECTION_RIGHT: 1, DIRECTION_LEFT: -1}


    def move_player(self, direction):
        (x, y) = self.player_pos
        dx = self.dir_mapping_x.get(direction, 0)
        dy = self.dir_mapping_y.get(direction, 0)
        if not self.wrap:
            x = np.clip(x + dx, 0, self.width-1)
            y = np.clip(y + dy, 0, self.height-1)
        else:
            x = (x + dx) % self.width
            y = (y + dy) % self.height
        self.player_pos = (x, y)


    def generate_objects(self):
        return {'player': self.player_pos}


class Visualizer(object):

    def __init__(self, game_logic, block_size=50, fps=30, use_gui=False):
        self.game_logic = game_logic
        self.block_size = block_size
        self.fps = fps
        self.use_gui = use_gui
        if self.use_gui:
            self.screen = pygame.display.set_mode((game_logic.width * block_size, game_logic.height * block_size))
        else:
            self.screen  = pygame.Surface((game_logic.width * block_size, game_logic.height * block_size))

        self.screen_pixels = np.zeros((block_size * game_logic.width, block_size * game_logic.height, 3), dtype=np.uint8)
        self.clock = pygame.time.Clock()
        self.action_mapping = {0: DIRECTION_NONE, 1: DIRECTION_LEFT, 2: DIRECTION_UP, 3: DIRECTION_DOWN, 4: DIRECTION_RIGHT}


    def draw_rect(self, (x, y), color):
        pygame.draw.rect(self.screen, color, (self.block_size*x, self.block_size*y, self.block_size, self.block_size), 0)


    def draw_screen(self):
        game_objects = self.game_logic.generate_objects()
        self.screen_pixels = self.generate_screen(game_objects)
        if self.use_gui:
            pygame.display.update()


    def generate_screen(self, game_objects):
        self.screen.fill(COLOR_WHITE)
        self.draw_rect(game_objects['player'], COLOR_BLUE)
        pixels = np.transpose(pygame.surfarray.array3d(self.screen), [1, 0, 2])[:, :, [2, 0, 1]]
        return pixels


    def generate_lowdim_screen(self, game_objects):
        (x, y) = game_objects['player']
        position = 4*y + x
        array = np.zeros([16], dtype=np.float32)
        array[position] = 1.
        return array

    def get_blue_shade(self, shade):
        combined_color = shade*np.array([0, 0, 255]) + (1-shade)*np.array([255, 255, 255])
        return tuple(combined_color.astype(np.uint8).tolist())

    def generate_screen_from_lowdim(self, lowdim):
        self.screen.fill(COLOR_WHITE)
        for i in range(16):
            y = i / 4
            x = i % 4
            self.draw_rect((x, y), self.get_blue_shade(lowdim[i]))
        return np.transpose(pygame.surfarray.array3d(self.screen), [1, 0, 2])[:, :, [2, 0, 1]]

    def perform_action(self, action_index, lowdim=False):
        if lowdim:
            old_screen = self.generate_lowdim_screen(self.game_logic.generate_objects())
        else:
            old_screen = self.screen_pixels.copy()
        direction = self.action_mapping[action_index]
        self.game_logic.move_player(direction)
        self.draw_screen()
        if lowdim:
            new_screen = self.generate_lowdim_screen(self.game_logic.generate_objects())
        else:
            new_screen = self.screen_pixels.copy()
        return old_screen, action_index, new_screen


    def run_with_human_player(self):
        while True:
            self.clock.tick(self.fps)
            dir = self.handle_human_input()
            self.game_logic.move_player(dir)
            self.draw_screen()


    def handle_human_input(self):
        dir = DIRECTION_NONE
        for event in pygame.event.get():
            if not hasattr(event, 'key'): continue
            if event.type == KEYDOWN:
                if event.key == K_RIGHT: dir = DIRECTION_RIGHT
                elif event.key == K_LEFT: dir = DIRECTION_LEFT
                elif event.key == K_UP: dir = DIRECTION_UP
                elif event.key == K_DOWN: dir = DIRECTION_DOWN
                elif event.key == K_ESCAPE: sys.exit(0)
        return dir



if __name__ == '__main__':
    game = GameLogic(width=5, height=5)
    vis = Visualizer(game, use_gui=False)
    vis.run_with_human_player()
