from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from collections import namedtuple
import numpy as np
import go_vncdriver
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
import config
import globalvar as GlobalVar
import argparse
import random
import os
import copy
import wgan_models.dcgan as dcgan
import wgan_models.mlp as mlp
import support_lib
import config
import subprocess
import time
import multiprocessing
import gan
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

class GanTrainer():
    """
    This thread runs gan training
    """
    def __init__(self):
        
        self.gan = gan.gan() # create gan
        self.last_load_time = time.time() # record last_load_time as initialize time

    def load_data(self):
        self.load_time = time.time() # get this load time
        if (self.load_time-self.last_load_time) >= config.gan_worker_com_internal:

            '''if it is time to load'''
            print('Try loading data...')
            try:
                data = np.load(config.datadir+'data.npz')['data'] # load data
                print('Data loaded: '+str(np.shape(data)))
                self.last_load_time = time.time() # record last load time
                self.gan.push_data(data) # push data to gan
                np.savez(config.datadir+'data.npz',
                         data=self.gan.empty_dataset)
            except Exception, e:
                print(str(Exception)+": "+str(e))

    def run(self):

        while True:

            '''keep running'''
            self.load_data()
            self.gan.train()

if __name__ == "__main__":
    trainer = GanTrainer()
    trainer.run()