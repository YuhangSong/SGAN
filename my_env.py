import numpy as np
import config
import time
def get_grid_observation(x, y):
    observation = np.ones((config.gan_size,config.gan_size))
    observation[int(x*(float(config.gan_size)/float(config.grid_size))):int((x+1)*(float(config.gan_size)/float(config.grid_size))),int(y*(float(config.gan_size)/float(config.grid_size))):int((y+1)*(float(config.gan_size)/float(config.grid_size)))] = 0.0
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
        if config.grid_type is 'normal':
            '''normal grid, reset to start point'''
            self.cur_x = 0
            self.cur_y = 0
        elif config.grid_type is 'simple_one_move':
            '''simple one move, reset to middle'''
            self.cur_x = config.grid_size/2-1
            self.cur_y = config.grid_size/2-1
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

            '''randomlize action'''
            action_dic = range(config.action_space)

            action_dic_p = np.zeros((config.action_space))
            for i in range(len(action_dic_p)):
                distance = abs(i-self.action)
                if distance > (config.action_space/2):
                    distance = distance - (config.action_space/2)
                if config.grid_random_type is 'discount':
                    action_dic_p[i] = config.grid_action_random_discounter**(distance)
                elif config.grid_random_type is 'russell':
                    if distance==0:
                        action_dic_p[i]=0.25
                    elif distance==1:
                        action_dic_p[i]=0.25
                    elif distance==2:
                        action_dic_p[i]=0.25
                    else:
                        print('error')
                        print(s)
            action_dic_p = action_dic_p / np.sum(action_dic_p)

            self.action = np.random.choice(a=action_dic, 
                                           p=action_dic_p)

            self.action = int(self.action)

            if self.action is 0:
                self.cur_x += 1
            elif self.action is 1:
                self.cur_x -= 1
            elif self.action is 2:
                self.cur_y += 1
            elif self.action is 3:
                self.cur_y -= 1
            else:
                print(s)

            if self.cur_x >= config.grid_size:
                self.cur_x = config.grid_size-1
            if self.cur_y >= config.grid_size:
                self.cur_y = config.grid_size-1
            if self.cur_x < 0:
                self.cur_x = 0
            if self.cur_y < 0:
                self.cur_y = 0

            time.sleep(config.lower_env_worker)

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