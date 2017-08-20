import numpy as np
import visualizer
import cv2
from random import choice
import itertools
import numpy as np
import torch

IMAGE = 0
VECTOR = 1
BLOCK_SIZE = 5
ACCEPT_GATE = 0.1

class Walk1D(object):

    def __init__(self, length, prob_left, mode):
        assert 0 <= prob_left <= 1
        assert mode in [IMAGE, VECTOR]
        self.LEFT, self.RIGHT = 0, 1
        self.mode = mode
        self.n = length
        self.p = prob_left
        self.state = np.random.randint(0, length)
        self.visualizer = visualizer.Visualizer(BLOCK_SIZE, 1, self.n,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE})
        self.cleaner_function = clean_entry_01_vec

    def get_vector_size(self):
        return self.n

    def get_all_possible_start_states(self):
        return [self.get_state(pos) for pos in range(self.n)]

    def get_transition_probs(self, state_vec):
        all_possible = self.get_all_possible_start_states()
        state_pos = np.argmax(state_vec)
        prob_dict = {tuple(state): 0. for state in all_possible}
        prob_dict[tuple(self.get_state(self.update_state(state_pos, self.LEFT)))] = self.p
        prob_dict[tuple(self.get_state(self.update_state(state_pos, self.RIGHT)))] = 1 - self.p
        return prob_dict


    def set_state(self, state):
        self.state = state

    def get_state(self, pos=None):
        if pos is None:
            pos = self.state
        onehot = [0] * self.n
        onehot[pos] = 1
        if self.mode == IMAGE:
            return self.visualizer.make_screen([onehot])
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


class BitFlip1D(object):

    def __init__(self, length, mode):
        assert mode in [IMAGE, VECTOR]
        self.mode = mode
        self.n = length
        self.state = np.random.randint(0, 2, size=(1, self.n))
        self.visualizer = visualizer.Visualizer(BLOCK_SIZE, 1, self.n,
                                                {0: visualizer.WHITE,
                                                 1: visualizer.BLUE})
        self.cleaner_function = clean_entry_01_vec

    def get_vector_size(self):
        return self.n

    def get_all_possible_start_states(self):
        return [np.array(x) for x in itertools.product([0, 1], repeat=self.n)]

    def get_transition_probs(self, state_vec):
        state_internal_repr = np.reshape(state_vec, [1, self.n])
        possible_states = self.get_all_possible_start_states()
        prob_dict = {tuple(state): 0. for state in possible_states}
        for i in range(self.n):
            prob_dict[tuple(self.get_state(self.update_state(state_internal_repr, i)))] = 1./self.n
        return prob_dict

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = np.random.randint(0, 2, size=(1, self.n))


    def get_state(self, state=None):
        if state is None:
            state = self.state
        if self.mode == IMAGE:
            return self.visualizer.make_screen(state)
        else:
            return state.copy()[0, :]

    def update_state(self, state, bit_to_flip):
        state = state.copy()
        state[0, bit_to_flip] = 1 - state[0, bit_to_flip]
        return state

    def update(self):
        bit_to_flip = np.random.randint(0, self.n)
        self.state = self.get_state(self.update_state(self.state, bit_to_flip))
        return self.state

class Walk2D(object):

    def __init__(self, width, height, prob_dirs, obstacle_pos_list, mode, should_wrap):
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

    def get_vector_size(self):
        return self.h * self.w

    def get_all_possible_start_states(self):
        return [self.get_state((x, y)) for (x, y) in self.non_obstacle_squares]

    def state_vector_to_position(self, state_vector):

        if self.mode==IMAGE:

            '''detect agent opsition from image'''

            agent_channel_should_be = [255,0,0]

            agent_count = 0
            for x in range(self.w):
                for y in range(self.h):
                    valid_on_channel = True
                    for c in range(3):
                        pixel_value_mean_on_channel = np.mean(state_vector[y*BLOCK_SIZE:(y+1)*BLOCK_SIZE,x*BLOCK_SIZE:(x+1)*BLOCK_SIZE,c])
                        if abs(pixel_value_mean_on_channel-agent_channel_should_be[c]) >= (255.0*ACCEPT_GATE):
                            valid_on_channel = False
                            break
                    if valid_on_channel:
                        pos = (x,y)
                        agent_count += 1

            if agent_count==1:
                return pos

            else:
                return 'bad state'

        elif self.mode==VECTOR:
            state_2d = np.reshape(state_vector, [self.h, self.w])
            for y in range(state_2d.shape[0]):
                for x in range(state_2d.shape[1]):
                    if state_2d[y, x] == 1:
                        return (x, y)
            raise Exception('No agent found in vector')
        else:
            raise Exception('Not a valid mode')

    def get_transition_probs(self, state_vector=None, state_pos=None):

        '''
        Yuhang: since I donot know your logic here, I thing using
        state_vector to detect state position is a waste of compution.
        So I just add a parameter of state_pos
        '''

        if self.mode==VECTOR:

            '''
            Yuhang: I have difficulty understanding the logicol here,
            I will leave these code here and implement the evaluation 
            of image in the next 'elif' block. If you wish, you can 
            integrate these two parts of the code.
            '''
            state_pos = self.state_vector_to_position(state_vector)
            all_possible = self.get_all_possible_start_states()
            prob_dict = {tuple(state): 0.0 for state in all_possible}
            prob_dict[tuple(self.get_state(self.update_state(state_pos, self.UP)))] += 0.8
            prob_dict[tuple(self.get_state(self.update_state(state_pos, self.LEFT)))] += 0.1
            prob_dict[tuple(self.get_state(self.update_state(state_pos, self.RIGHT)))] += 0.1
            prob_dict[tuple(self.get_state(self.update_state(state_pos, self.DOWN)))] += 0.0
            return prob_dict

        elif self.mode==IMAGE:

            prob_dict = {}
            for action_i in range(len(self.action_dic)):
                prob_dict[str(self.update_state(state_pos, self.action_dic[action_i]))] = self.prob_dirs[action_i]
            return prob_dict


    def reset(self):
        self.x_pos, self.y_pos = choice(self.non_obstacle_squares)


    def set_state(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        assert (x_pos, y_pos) not in self.obstacle_pos_list

    def get_state(self, state=None):
        if state is None:
            state = self.x_pos, self.y_pos
        x_pos, y_pos = state
        array = np.zeros((self.h, self.w), dtype=np.uint8)
        array[y_pos, x_pos] = 1
        for obs_x, obs_y in self.obstacle_pos_list:
            array[obs_y, obs_x] = 2
        if self.mode == IMAGE:
            return self.visualizer.make_screen(array)
        else:
            return np.reshape(array, [-1])

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
    l1 = 0.
    combined_keyset = set(dist1.keys()).union(set(dist2.keys()))
    for key in combined_keyset:
        l1 += np.abs(dist1.get(key, 0) - dist2.get(key, 0))
    return l1

def evaluate_domain(domain, s1_state, s2_samples):

    if domain.mode==VECTOR:
        '''
        Yuhang: I have difficulty understanding the logicol here,
        I will leave these code here and implement the evaluation 
        of image in the next 'elif' block. If you wish, you can 
        integrate these two parts of the code.
        '''
        s1_state = np.array(s1_state)
        true_distribution = domain.get_transition_probs(s1_state)
        sample_counts = {}
        bad_count = 0
        good_count = 0
        for b in range(np.shape(s2_samples)[0]):
            sample = s2_samples[b]
            determined_state = determine_transition(domain, sample)
            if determined_state == 'bad state':
                bad_count += 1
            else:
                if tuple(determined_state) in sample_counts:
                    sample_counts[tuple(determined_state)] += 1
                else:
                    sample_counts[tuple(determined_state)] = 1
                good_count += 1
        sample_distribution = {state: sample_count / float(good_count) for state, sample_count in sample_counts.items()}
        return l1_distance(true_distribution, sample_distribution), good_count / float(good_count + bad_count)

    elif domain.mode==IMAGE:
        s1_pos = domain.state_vector_to_position(s1_state)
        true_distribution = domain.get_transition_probs(
            state_pos=s1_pos
        )
        bad_count = 0
        good_count = 0
        sample_distribution = {}
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

        print(sample_distribution)
        for key in sample_distribution.keys():
            sample_distribution[key] = sample_distribution[key] / float(good_count)

        return l1_distance(true_distribution, sample_distribution), good_count / float(good_count + bad_count)
                


