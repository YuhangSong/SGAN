import numpy as np
import visualizer
import cv2
from random import choice
import itertools
import numpy as np
import torch

'''
    This code is initially provided by Chris
'''

# three representations
SCALAR = 0
VECTOR = 1
IMAGE = 2

# for image, the size of every block in grid world
BLOCK_SIZE = 4

# for evaluation, with in this gate, the generated image well
# be counted as valid 
ACCEPT_GATE = 0.1

# some of the domain has too many starts state, this will limit
# the evaluation to parts of the starts state, instead of evaluating
# them all
LIMIT_START_STATE_TO = 50

# for the background of the grid world, it is gray
FEATURE_DISCOUNT = 0.5

class Walk1D(object):

    def __init__(self, length, prob_left, mode, fix_state=False):
        assert 0 <= prob_left <= 1
        assert mode in [IMAGE, VECTOR]
        self.LEFT, self.RIGHT = 0, 1
        self.action_dic = [self.LEFT, self.RIGHT]
        self.mode = mode
        self.fix_state = fix_state
        self.n = length
        self.p = prob_left
        self.prob_dirs = [self.p, (1-self.p)]
        self.state = np.random.randint(0, length)
        self.visualizer = visualizer.Visualizer(BLOCK_SIZE, 1, self.n,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE})
        self.cleaner_function = clean_entry_01_vec

    def get_vector_size(self):
        return self.n

    def get_all_possible_start_states(self):
        return [self.get_state(pos) for pos in range(self.n)]

    def get_transition_probs(self, state_pos, is_tabular=False):

        prob_dict = {}

        for action_i in range(len(self.action_dic)):

            key = str(self.update_state(state_pos, self.action_dic[action_i]))

            if prob_dict.has_key(key):
                prob_dict[key] += self.prob_dirs[action_i]
            else:
                prob_dict[key] = self.prob_dirs[action_i]

        return prob_dict


    def set_state(self, state):
        self.state = state

    def get_state(self, pos=None):
        if pos is None:
            pos = self.state
        onehot = [0] * self.n
        onehot[pos] = 1
        if self.mode == IMAGE:
            image = self.visualizer.make_screen([onehot])
            image = image[:,:,1:2]
            image = image / 255.0
            image = 1.0 - image
            return image
        else:
            return np.array(onehot)

    def update_state(self, state, direction):
        delta = -1 if direction == self.LEFT else 1
        new_state = np.clip(state+delta, 0, self.n-1)
        return new_state

    def reset(self):
        self.state = np.random.randint(0, self.n)

    def update(self):
        direction = self.LEFT if np.random.uniform(0, 1) < self.p else self.RIGHT
        self.state = self.update_state(self.state, direction)
        return self.get_state(self.state)

    def state_vector_to_position(self, state_vector, include_background=False):

        if self.mode==IMAGE:

            '''detect agent opsition from image'''

            agent_channel_should_be = [1.0]

            agent_count = 0
            for x in range(self.n):
                valid_on_channel = True
                for c in range(1):
                    pixel_value_mean_on_channel = np.mean(state_vector[:,x*BLOCK_SIZE:(x+1)*BLOCK_SIZE,c])
                    if abs(pixel_value_mean_on_channel-agent_channel_should_be[c]) >= (ACCEPT_GATE):
                        valid_on_channel = False
                        break
                if valid_on_channel:
                    pos = x
                    agent_count += 1

            if agent_count==1:
                return pos

            else:
                return 'bad state'

        elif self.mode==VECTOR:

            agent_channel_should_be = 1.0

            agent_count = 0
            for x in range(self.n):
                if abs(state_vector[x]-agent_channel_should_be) < 1.0*ACCEPT_GATE:
                    pos = x
                    agent_count += 1

            if agent_count==1:
                return pos

            else:
                return 'bad state'

class Walk2D(object):

    def __init__(self, width, height, prob_dirs, obstacle_pos_list, mode, should_wrap, fix_state=False, random_background=False):
        self.UP, self.DOWN, self.LEFT, self.RIGHT = 0, 1, 2, 3
        self.action_dic = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
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
        self.non_obstacle_squares = [(x, y) for x in range(self.w) for y in range(self.h)
                                     if (x, y) not in self.obstacle_pos_list]
        self.should_wrap = should_wrap
        self.mode = mode
        self.x_pos, self.y_pos = choice(self.non_obstacle_squares)
        self.visualizer = visualizer.Visualizer(BLOCK_SIZE, height, width,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE,
                                                 2: visualizer.BLACK})
        self.cleaner_function = clean_entry_012_vec
        self.fix_state = fix_state
        self.fix_state_to = (self.w/2, self.h/2)
        self.random_background = random_background

        self.reset_background()
        self.build_background_feature_mask()

    def reset_background(self):
        self.background_array = np.random.randint(
            2, 
            size=(self.h, self.w),
            dtype=np.uint8,
        )
        if not self.random_background:
            self.background_array = self.background_array*0.0
    
    def build_background_feature_mask(self):
        self.background_feature_mask = (1.0-self.visualizer.make_screen(self.background_array)[:,:,1:2]/255.0)
        for y in range(np.shape(self.background_feature_mask)[0]):
            self.background_feature_mask[y,:,:] = y%2

        self.background_feature_block = np.copy(self.background_feature_mask[0:BLOCK_SIZE,0:BLOCK_SIZE,:])
        for y in range(np.shape(self.background_feature_block)[0]):
            self.background_feature_block[y,:,:] = y%2
        self.background_feature_block = self.background_feature_block * FEATURE_DISCOUNT

        self.background_unfeature_block = np.copy(self.background_feature_block)
        self.background_unfeature_block = self.background_unfeature_block*0.0

        self.agent_block = np.copy(self.background_unfeature_block)
        self.agent_block = 1.0 - self.agent_block

    def set_fix_state(self,fix_state):
        self.fix_state = fix_state

    def get_vector_size(self):
        return self.h * self.w

    def get_all_possible_start_states(self):

        if self.fix_state:
            return [self.get_state(self.fix_state_to)]

        else:
            return [self.get_state((x, y)) for (x, y) in self.non_obstacle_squares]

    def state_vector_to_position(self, state_vector, include_background=False):

        if self.mode==SCALAR:

            agent_count = 0
            for x in range(self.w):
                for y in range(self.h):
                    if np.abs(float(x)/self.w-state_vector[0])<(1.0/self.w*ACCEPT_GATE) and np.abs(float(y)/self.h-state_vector[1])<(1.0/self.h*ACCEPT_GATE):
                        pos = (x,y)
                        agent_count += 1

            if agent_count==1:
                return pos

            else:
                return 'bad state'

        elif self.mode==VECTOR:
            raise Exception('error')

        elif self.mode==IMAGE:

            '''detect agent opsition from image'''
            agent_count = 0
            for x in range(self.w):
                for y in range(self.h):
                    block = state_vector[y*BLOCK_SIZE:(y+1)*BLOCK_SIZE,x*BLOCK_SIZE:(x+1)*BLOCK_SIZE,:]
                    close_to_agent = np.mean(np.abs(block-self.agent_block))
                    if close_to_agent <= ACCEPT_GATE:
                        '''if agent is here'''
                        pos = (x,y)
                        agent_count += 1
                        self.background_array[y,x] = 255
                    else:
                        '''if agent is not here'''
                        if self.random_background:
                            close_to_feature_background = np.mean(np.abs(block-self.background_feature_block))
                            close_to_unfeature_background = np.mean(np.abs(block-self.background_unfeature_block))
                            if include_background:
                                '''if start state, set the self.background_array'''        
                                if close_to_feature_background <= ACCEPT_GATE:
                                    self.background_array[y,x] = 1
                                elif close_to_unfeature_background <= ACCEPT_GATE:
                                    self.background_array[y,x] = 0
                                else:
                                    return 'bad state'
                            else:
                                '''see if generate background right'''
                                if self.background_array[y,x]==1:
                                    if not (close_to_feature_background <= ACCEPT_GATE):
                                        return 'bad state'
                                elif self.background_array[y,x]==0:
                                    if not (close_to_unfeature_background <= ACCEPT_GATE):
                                        return 'bad state'
                                elif self.background_array[y,x]==255:
                                    pass
                                else:
                                    print(self.background_array[y,x])
                                    raise Exception('s')


            if agent_count==1:
                return pos

            else:
                return 'bad state'

        else:
            raise Exception('Not a valid mode')

    def get_state_str(self, state_vector):
        return str((state_vector/FEATURE_DISCOUNT).astype(int)[:,:,0])

    def get_transition_probs(self, state_vector=None, state_pos=None, is_tabular=False):

        prob_dict = {}

        for action_i in range(len(self.action_dic)):

            if is_tabular:
                key = self.get_state_str(self.get_state(self.update_state(state_pos, self.action_dic[action_i])))
            else:
                key = str(self.update_state(state_pos, self.action_dic[action_i]))

            if prob_dict.has_key(key):
                prob_dict[key] += self.prob_dirs[action_i]
            else:
                prob_dict[key] = self.prob_dirs[action_i]

        return prob_dict

    def reset(self):
        if not self.fix_state:
            self.x_pos, self.y_pos = choice(self.non_obstacle_squares)

        else:
            self.x_pos, self.y_pos = self.fix_state_to

        self.reset_background()

    def set_state(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        assert (x_pos, y_pos) not in self.obstacle_pos_list

    def get_screen(self,x):
        return (1.0-self.visualizer.make_screen(x)[:,:,1:2]/255.0)

    def get_state(self, state=None):
        if state is None:
            state = self.x_pos, self.y_pos
        x_pos, y_pos = state
        array = np.zeros((self.h, self.w), dtype=np.uint8)
        array[y_pos, x_pos] = 1
        for obs_x, obs_y in self.obstacle_pos_list:
            array[obs_y, obs_x] = 2

        if self.mode == SCALAR:
            return np.array([float(x_pos)/float(self.w),float(y_pos)/float(self.h)])

        elif self.mode == VECTOR:
            return np.reshape(array, [-1])

        elif self.mode == IMAGE:
            if self.obstacle_pos_list==[]:
                image_agent = self.get_screen(array)
                if self.random_background:
                    image_background = self.get_screen(self.background_array)
                    image_background = image_background*self.background_feature_mask*FEATURE_DISCOUNT
                    image_background_no_agent = image_background*(1.0-image_agent)
                    image = image_agent + image_background_no_agent
                else:
                    image = image_agent
            else:
                if self.random_background:
                    raise Exception('s')
                image = self.visualizer.make_screen(array)
                image = image / 255.0
                image = 1.0 - image
                image_obst = image[:,:,0:1]
                image_agent = image[:,:,1:2] - image_obst
                image = image_agent + image_obst * 0.5
            return image
        
        else:
            print(sss)

    def update_state(self, (x, y), action):
        (delta_x, delta_y) = self.action_delta_mapping[action]
        if self.should_wrap:
            new_x_pos = (x+delta_x) % self.w
            new_y_pos = (y+delta_y) % self.h
        else:
            new_x_pos = np.clip(x+delta_x, 0, self.w-1)
            new_y_pos = np.clip(y+delta_y, 0, self.h-1)
        if (new_x_pos, new_y_pos) not in self.obstacle_pos_list:
            return (new_x_pos, new_y_pos)
        else:
            return (x, y)

    def update(self):
        action = np.random.choice([self.UP, self.DOWN, self.LEFT, self.RIGHT], p=self.prob_dirs)
        self.x_pos, self.y_pos = self.update_state((self.x_pos, self.y_pos), action)
        return self.get_state((self.x_pos, self.y_pos))

def clean_entry_01(entry):
    if np.abs(entry - 0) <= ACCEPT_GATE:
        return 0
    elif np.abs(entry - 1) <= ACCEPT_GATE:
        return 1
    else:
        return -1

def clean_entry_012(entry):
    if np.abs(entry - 0) <= ACCEPT_GATE:
        return 0
    elif np.abs(entry - 1) <= ACCEPT_GATE:
        return 1
    elif np.abs(entry - 2) <= ACCEPT_GATE:
        return 2
    else:
        return -1

clean_entry_01_vec = np.vectorize(clean_entry_01, [np.int32])
clean_entry_012_vec = np.vectorize(clean_entry_012, [np.int32])

def determine_transition(domain, sample):
    cleaned_sample = domain.cleaner_function(sample)
    if np.min(cleaned_sample) == -1:
        return 'bad state'
    else:
        return cleaned_sample

def l1_distance(dist1, dist2):
    l1 = 0.0
    combined_keyset = set(dist1.keys()).union(set(dist2.keys()))
    for key in combined_keyset:
        l1 += np.abs(dist1.get(key, 0) - dist2.get(key, 0))
    return l1

def evaluate_domain(domain, s1_state, s2_samples, is_tabular=False):

    s1_pos = domain.state_vector_to_position(
        s1_state,
        include_background = True,
    )
    # print(s1_pos)
    # print(domain.background_array)
    # print(s)

    true_distribution = domain.get_transition_probs(
        state_pos=s1_pos,
        is_tabular=is_tabular,
    )
    # print(true_distribution)
    bad_count = 0
    good_count = 0
    sample_distribution = {}

    if is_tabular:

        for s2_sample in s2_samples.next_dic.keys():
            good_count += s2_samples.next_dic[s2_sample]
            try:
                sample_distribution[s2_sample] = sample_distribution[str(s2_sample_pos)] + s2_samples.next_dic[s2_sample]
            except Exception as e:
                sample_distribution[s2_sample] = s2_samples.next_dic[s2_sample]

    else:
        for b in range(np.shape(s2_samples)[0]):
            s2_sample_pos = domain.state_vector_to_position(s2_samples[b])
            if s2_sample_pos=='bad state':
                bad_count += 1
            else:
                good_count += 1
                try:
                    sample_distribution[str(s2_sample_pos)] = sample_distribution[str(s2_sample_pos)] + 1.0
                except Exception as e:
                    sample_distribution[str(s2_sample_pos)] = 1

    if good_count>0.0:

        for key in sample_distribution.keys():
            sample_distribution[key] = sample_distribution[key] / float(good_count)
        print('----------------------------------------------')
        print('Start: '+str(s1_pos))
        print('True: '+str(true_distribution))
        print('Sample: '+str(sample_distribution))
        L1 = l1_distance(true_distribution, sample_distribution)
        AC = good_count / float(good_count + bad_count)
        return L1, AC
    else:

        return 2.0, 0.0

            


