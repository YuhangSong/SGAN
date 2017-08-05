import numpy as np
import config
import time
def get_grid_2d_observation(x, y):
    observation = np.ones((config.gan_size,config.gan_size))
    observation[x*(config.gan_size/config.grid_size):(x+1)*(config.gan_size/config.grid_size),y*(config.gan_size/config.grid_size):(y+1)*(config.gan_size/config.grid_size)] = 0.0
    observation = np.expand_dims(a=observation,
                                 axis=0)
    observation = np.concatenate((observation,observation,observation),
                                 axis=0)
    return observation

def get_grid_1d_observation(x):
    observation = np.zeros((3,config.gan_size,config.gan_size))
    observation[0:1,0:1,x:(x+1)] = 1.0
    return observation

class env():
    def __init__(self):

        self.will_reset = False
        self.step = 0

        #####################################################
        if config.grid_type is '1d_fall':
            self.step_limit = -1
        elif config.grid_type is '2d_one_move' or config.grid_type is '2d_action_random' or config.grid_type is '2d_action':
            self.step_limit = config.grid_size*4
        #####################################################

    def reset(self):
        ################# reset state ################
        if config.grid_type is '1d_fall':
            self.cur_x = 0
        elif config.grid_type is '2d_one_move':
            '''simple one move, reset to middle'''
            self.cur_x = config.grid_size/2-1
            self.cur_y = config.grid_size/2-1
        elif config.grid_type is '2d_action_random' or config.grid_type is '2d_action':
            '''normal grid, reset to start point'''
            self.cur_x = 0
            self.cur_y = 0
        ##############################################

        self.step = 0
        self.done = True
        self.reward = 0.0
        self.will_reset = False

    def if_win(self):
        ################# if win #####################
        if config.grid_type is '1d_fall':
            win = False
        ##############################################
        return win

    def update_observation(self):
        ################# update observation #####################
        if config.grid_type is '1d_fall':
            self.observation = get_grid_1d_observation(self.cur_x)
        elif config.grid_type is '2d_one_move' or config.grid_type is '2d_action_random' or config.grid_type is '2d_action':
            self.observation = get_grid_2d_observation(self.cur_x,self.cur_y)
        ##########################################################

    def get_initial_observation(self):
        self.reset()
        self.update_observation()
        return self.observation

    def act(self, action):

        self.action = action
        self.step += 1
        
        if self.will_reset:
            self.reset()
        else:
            self.done = False

            ########################## update state #########################

            '''randomlize action'''
            action_dic = range(config.action_space)

            action_dic_p1 = np.zeros((config.action_space))
            action_dic_p2 = np.zeros((config.action_space))

            for i in range(len(action_dic_p1)):
                distance = abs(i-80)
                if distance > (config.action_space/2):
                    distance = config.action_space - distance                
                action_dic_p1[i] = 0.7**distance
            for i in range(len(action_dic_p2)):
                distance = abs(i-40)
                if distance > (config.action_space/2):
                    distance = config.action_space - distance                
                action_dic_p2[i] = 0.7**distance

            action_dic_p = action_dic_p1 * 0.7 + action_dic_p2 * 0.3

            action_dic_p = action_dic_p / np.sum(action_dic_p)

            self.action = np.random.choice(a=action_dic, 
                                           p=action_dic_p)

            self.action = int(self.action)

            if config.grid_type is '1d_fall':
                self.cur_x = self.action
            else:
                if self.action is 0:
                    self.cur_x += 1
                elif self.action is 1:
                    self.cur_x -= 1
                elif self.action is 2:
                    self.cur_y += 1
                elif self.action is 3:
                    self.cur_y -= 1

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
                if self.step_limit > 0:
                    if self.step > self.step_limit:
                        self.will_reset = True

            if config.grid_type is 'simple_one_move':
                '''simple one move, overwrite'''
                if self.step <=2:
                    self.cur_x = config.grid_size/2-1
                    self.cur_y = config.grid_size/2-1
                    self.will_reset = False
                else:
                    self.will_reset = True

        self.update_observation()

        return self.observation, self.reward, self.done