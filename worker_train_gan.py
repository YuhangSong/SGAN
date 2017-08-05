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
        
        if (time.time()-self.last_load_time) >= config.gan_worker_com_internal:

            '''if it is time to load'''
            print('Try loading data...')

            data = None
            try:
                data = np.load(config.datadir+'data.npz')['data'] # load data
                print('Load data: '+str(np.shape(data)))
            except Exception, e:
                print('Load failed')
                # print(str(Exception)+": "+str(e))

            '''delete any way too avoid futher bug'''
            subprocess.call(["rm", "-r", "-f", config.datadir+'data.npz'])

            if data is not None:
                self.last_load_time = time.time() # record last load time
                self.gan.push_data(data) # push data to gan

    def run(self):

        while True:

            need_load_data = True
            if config.gan_dataset_full_no_update:
                if self.gan.if_dataset_full():
                    need_load_data = False

            '''keep running'''
            if need_load_data:
                self.load_data()
                
            self.gan.train()


if __name__ == "__main__":
    trainer = GanTrainer()
    trainer.run()