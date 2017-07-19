import numpy as np
import config
def get_grid_observation(x, y):
    observation = np.zeros((config.gan_size,config.gan_size))
    observation[x*(config.gan_size/config.grid_size):(x+1)*(config.gan_size/config.grid_size),y*(config.gan_size/config.grid_size):(y+1)*(config.gan_size/config.grid_size)] = 1.0
    observation = np.expand_dims(a=observation,
                                 axis=0)
    observation = np.concatenate((observation,observation,observation),
                                 axis=0)
    return observation
class env():
    def __init__(self):
        self.will_reset = False
        self.step = 0
        self.step_limit = config.grid_size*4

    def reset(self):
        ################# reset state ################
        self.cur_x = 0
        self.cur_y = 0
        ##############################################
        self.step = 0
        self.done = True
        self.reward = 0.0
        self.will_reset = False

    def if_win(self):
        ################# if win #####################
        win = self.cur_x is config.grid_target_x and self.cur_y is config.grid_target_y
        ##############################################
        return win

    def update_observation(self):
        ################# update observation #####################
        self.observation = get_grid_observation(self.cur_x,self.cur_y)
        ##########################################################

    def get_initial_observation(self):
        self.reset()
        self.update_observation()
        print('--------------------------------------')
        return self.observation

    def act(self, action):

        self.action = action
        self.step += 1
        
        if self.will_reset:
            self.reset()
        else:
            self.done = False

            ########################## update state #########################
            if self.action is 0:
                self.cur_x += 1
            elif self.action is 1:
                self.cur_x -= 1
            elif self.action is 2:
                self.cur_y += 1
            elif self.action is 3:
                self.cur_y -= 1
            else:
                print(exit)

            if self.cur_x >= config.grid_size:
                self.cur_x = config.grid_size-1
            if self.cur_y >= config.grid_size:
                self.cur_y = config.grid_size-1
            if self.cur_x < 0:
                self.cur_x = 0
            if self.cur_y < 0:
                self.cur_y = 0

            ###########################################################

            '''judging done'''
            self.will_reset = False
            if self.if_win():
                self.reward = 1.0
                self.will_reset = True
            else:
                self.reward = 0.0
                if self.step > self.step_limit:
                    self.will_reset = True

        self.update_observation()

        return self.observation, self.reward, self.done