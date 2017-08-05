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
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import visdom
vis = visdom.Visdom()
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

class gan():
    """
    This thread runs gan training
    """
    def __init__(self):
        
        '''config'''
        self.cuda = True
        self.ngpu = config.gan_ngpu
        self.nz = config.gan_nz
        self.ngf = 64
        self.ndf = 64
        self.nc = config.gan_nc
        self.n_extra_layers = 0
        self.imageSize = config.gan_size
        self.lrD = 0.00005
        self.lrC = 0.00005
        self.lrG = 0.00005
        self.batchSize = config.gan_batchsize
        self.DCiters_ = 5
        self.clamp_lower = -0.01
        self.clamp_upper = 0.01
        self.experiment = config.logdir
        self.dataset_limit = config.gan_dataset_limit
        self.aux_size = config.gan_aux_size
        
        self.empty_dataset_with_aux = np.zeros((0, 5, self.nc, self.imageSize, self.imageSize))

        '''random seed for torch'''
        self.manualSeed = random.randint(1, 10000) # fix seed
        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not self.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        '''custom weights initialization called on netG and netD'''
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        '''create models'''
        self.netG = dcgan.DCGAN_G().cuda()
        self.netD = dcgan.DCGAN_D().cuda()

        '''init models'''
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.load_models()

        '''print the models'''
        print(self.netD)
        self.netG.deconv_layer_1.weight.data.copy_(self.netG.deconv_layer_0.weight.data)
        self.netG.deconv_layer_2.weight.data.copy_(self.netG.deconv_layer_0.weight.data)
        self.netG.deconv_layer_3.weight.data.copy_(self.netG.deconv_layer_0.weight.data)
        print(self.netG)

        # noise
        self.noise = torch.FloatTensor(self.batchSize, self.aux_size).cuda()

        # constent
        self.one = torch.FloatTensor([1]).cuda()
        self.mone = self.one * -1

        '''dataset intialize'''
        if config.grid_type is '1d_fall':
            self.dataset_image = torch.FloatTensor(np.zeros((1, 4, config.action_space)))
        else:
            self.dataset_image = torch.FloatTensor(np.zeros((1, 4, self.nc, self.imageSize, self.imageSize)))

        self.dataset_aux = torch.FloatTensor(np.zeros((1, self.aux_size)))

        '''recorders'''
        self.recorder_cur_errD = torch.FloatTensor([0])
        self.recorder_target_mse = torch.FloatTensor([0])
        self.recorder_loss_mse = torch.FloatTensor([0])
        self.recorder_loss_g = torch.FloatTensor([0])
        self.recorder_loss_g_mse = torch.FloatTensor([0])

        self.indexs_selector = torch.LongTensor(self.batchSize)

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

        self.iteration_i = 0
        self.last_save_model_time = 0
        self.last_save_image_time = 0

        self.target_errD = 0.001
        self.target_mse_p = 0.2
        self.target_mse = 1.0

        self.mse_loss_model = torch.nn.MSELoss(size_average=True)

    def train(self):
        """
        train one iteraction
        """

        if self.dataset_image.size()[0] >= self.batchSize:

            '''only train when have enough dataset'''
            print('Train on dataset: '+str(int(self.dataset_image.size()[0])))

            ########################################################################################################
            ############################################ Update D network ##########################################
            ########################################################################################################

            for p in self.netD.parameters():
                p.requires_grad = True

            # if (self.iteration_i < 25) or (self.iteration_i % 500 == 0):
            #     DCiters = 100
            # else:
            #     DCiters = self.DCiters_
            DCiters = self.DCiters_

            j = 0
            while j < DCiters:

                j += 1

                # clamp parameters to a cube
                for p in self.netD.parameters():
                    p.data.clamp_(self.clamp_lower, self.clamp_upper)

                ################################## load a trained batch #####################################
                # generate indexs
                indexs = self.indexs_selector.random_(0,self.dataset_image.size()[0])

                # indexing image
                self.state_prediction_gt_raw = torch.index_select(self.dataset_image,0,indexs).cuda()
                self.state = self.state_prediction_gt_raw.narrow(1,0,config.state_depth)
                self.prediction_gt_raw = self.state_prediction_gt_raw.narrow(1,config.state_depth,1)
                
                # indexing aux
                self.aux = torch.index_select(self.dataset_aux,0,indexs).cuda()
                #############################################################################################

                if config.using_r:
                    ################################## go through r #############################################
                    self.prediction_gt = self.netG( input_image = Variable( self.state_prediction_gt_raw.narrow(1,1,config.state_depth),
                                                                            volatile = True),
                                                    input_aux   = Variable( self.aux,
                                                                            volatile = True),
                                                    input_noise = Variable( self.noise.normal_(0, 1),
                                                                            volatile = True)
                                                    ).narrow(1,config.state_depth-1,1).data
                    ##############################################################################################
                else:
                    self.prediction_gt = self.prediction_gt_raw
                
                self.state_prediction_gt = torch.cat([self.state,self.prediction_gt],1)
                ################################# get fake ###################################################
                self.prediction =    self.netG( input_image = Variable( self.state,
                                                                        volatile = True),
                                                input_aux   = Variable( self.aux,
                                                                        volatile = True),
                                                input_noise = Variable( self.noise.normal_(0, 1),
                                                                        volatile = True)
                                                ).narrow(1,config.state_depth,1).data
                self.state_prediction = torch.cat([self.state, self.prediction], 1)
                ##############################################################################################
                    

                ######################################## train D #############################################
                self.netD.zero_grad()

                ################# train D with real #################
                errD_real_v =        self.netD( input_image = Variable( self.state_prediction_gt),
                                                input_aux   = Variable( self.aux)
                                                ).mean(0).view(1)
                errD_real_v.backward(self.one)
                #####################################################

                ################# train D with fake #################
                errD_fake_v =        self.netD( input_image = Variable( self.state_prediction),
                                                input_aux   = Variable( self.aux)
                                                ).mean(0).view(1)
                errD_fake_v.backward(self.mone)
                #####################################################

                cur_errD = (errD_real_v - errD_fake_v).data.abs()
                self.recorder_cur_errD = torch.cat([self.recorder_cur_errD,cur_errD.cpu()],0)
                self.recorder_cur_errD_mid_numpy = scipy.signal.medfilt(self.recorder_cur_errD.numpy(),25)

                self.optimizerD.step()
                ##############################################################################################

            ######################################## control target mse #################################
            if self.recorder_cur_errD_mid_numpy[-1] < self.target_errD:
                self.target_mse = self.target_mse - self.target_mse * self.target_mse_p
            else:
                self.target_mse = self.target_mse + self.target_mse * self.target_mse_p
            if not config.using_r:
                self.target_mse = 1.0
            self.target_mse = np.clip(self.target_mse,0.0,1.0)
            self.recorder_target_mse = torch.cat([self.recorder_target_mse,torch.FloatTensor([self.target_mse])],0)
            ##############################################################################################

            ########################################################################################################
            ######################################## End of Update D network #######################################
            ########################################################################################################

            ########################################################################################################
            ########################################### Update G network ###########################################
            ########################################################################################################

            for p in self.netD.parameters():
                p.requires_grad = False

            self.netG.zero_grad()

            x =  self.netG( input_image = Variable( self.state),
                            input_aux   = Variable( self.aux),
                            input_noise = Variable( self.noise.normal_(0, 1))
                            )
            self.stater_v = x.narrow(1,0,config.state_depth)
            self.prediction_v = x.narrow(1,config.state_depth,1)

            loss_mse_v = self.mse_loss_model(self.stater_v, Variable(self.state))
            self.recorder_loss_mse = torch.cat([self.recorder_loss_mse,loss_mse_v.data.cpu()],0)

            if loss_mse_v.data.cpu().numpy()[0] > self.target_mse:

                loss_g_mse_v = loss_mse_v
                self.recorder_loss_g_mse = torch.cat([self.recorder_loss_g_mse,torch.FloatTensor([1.0])],0)

            else:

                loss_g_v = self.netD(   input_image = torch.cat([Variable(self.state), self.prediction_v], 1),
                                        input_aux   = Variable(self.aux)
                                        ).mean(0).view(1)
                self.recorder_loss_g = torch.cat([self.recorder_loss_g,loss_g_v.data.cpu()],0)

                loss_g_mse_v = loss_g_v
                self.recorder_loss_g_mse = torch.cat([self.recorder_loss_g_mse,torch.FloatTensor([-1.0])],0)

            loss_g_mse_v.backward(self.one)

            self.optimizerG.step()
            ########################################################################################################
            ######################################## End of Update G network #######################################
            ########################################################################################################


            ########################################################################################################
            ############################################ One Iteration ### #########################################
            ########################################################################################################

            '''log result'''
            print(  '[iteration_i:%d]'
                    %(self.iteration_i))

            '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image_time) > config.gan_save_image_internal:

                multiple_one_state = torch.cat([self.state[0:1]]*self.batchSize,0)
                multiple_one_aux = torch.cat([self.aux[0:1]]*self.batchSize,0)

                self.prediction =  self.netG(   input_image = Variable( multiple_one_state),
                                                input_aux   = Variable( self.aux),
                                                input_noise = Variable( self.noise.normal_(0, 1))
                                                ).narrow(1,config.state_depth,1).data

                def to_one_hot(x):
                    x = x.squeeze(1)
                    x = x.cpu()
                    for w in range(x.size()[0]):
                        l = x[w].numpy()
                        max_ = np.max(l)
                        l[l<max_] = 0.0
                        l[l>=max_] = 1.0
                        l = torch.FloatTensor(l)
                        x[w] = l
                    return x
                
                self.prediction = to_one_hot(self.prediction)
                self.prediction_gt_raw = to_one_hot(self.prediction_gt_raw)
                self.prediction_gt = to_one_hot(self.prediction_gt)

                try:
                    self.recorder_prediction_gt_raw_heatmap = torch.cat([self.recorder_prediction_gt_raw_heatmap,self.prediction_gt_raw.mean(0)],0)
                except Exception, e:
                    self.recorder_prediction_gt_raw_heatmap = self.prediction_gt_raw.mean(0)
                self.heatmap(self.recorder_prediction_gt_raw_heatmap, 'recorder_prediction_gt_raw_heatmap')
                
                try:
                    self.recorder_prediction_gt_heatmap = torch.cat([self.recorder_prediction_gt_heatmap,self.prediction_gt.mean(0)],0)
                except Exception, e:
                    self.recorder_prediction_gt_heatmap = self.prediction_gt.mean(0)
                self.heatmap(self.recorder_prediction_gt_heatmap, 'recorder_prediction_gt_heatmap')

                try:
                    self.recorder_prediction_heatmap = torch.cat([self.recorder_prediction_heatmap,self.prediction.mean(0)],0)
                except Exception, e:
                    self.recorder_prediction_heatmap = self.prediction.mean(0)
                self.heatmap(self.recorder_prediction_heatmap, 'recorder_prediction_heatmap')

                self.line(self.recorder_cur_errD,'recorder_cur_errD')
                self.line(torch.FloatTensor(self.recorder_cur_errD_mid_numpy),'recorder_cur_errD_mid_numpy')
                self.line(self.recorder_target_mse,'recorder_target_mse')
                
                self.line(self.recorder_loss_mse,'recorder_loss_mse')

                self.line(self.recorder_loss_g,'recorder_loss_g')

                self.line(self.recorder_loss_g_mse,'recorder_loss_g_mse')

                self.last_save_image_time = time.time()

            self.iteration_i += 1

            ########################################################################################################
            ########################################## End One in Iteration  #######################################
            ########################################################################################################

        else:

            '''dataset not enough'''
            print('Dataset not enough: '+str(int(self.dataset_image.size()[0])))

            time.sleep(config.gan_worker_com_internal)

    def push_data(self, data):
        """
        push data to dataset which is a torch tensor
        """

        if np.shape(data)[0] is 0:
            return

        data = torch.FloatTensor(data)

        if config.grid_type is '1d_fall':
            data_image = data[:,0:4,0,0,0:config.action_space]
        else:
            data_image = data[:,0:4,:,:,:]

        data_aux = data[:,4:5,0:1,0:1,0:self.aux_size]
        data_aux = torch.squeeze(data_aux,1)
        data_aux = torch.squeeze(data_aux,1)
        data_aux = torch.squeeze(data_aux,1)

        if self.dataset_image.size()[0] <= 1:
            self.dataset_image = data_image
            self.dataset_aux = data_aux
        else:

            self.dataset_image = torch.cat(seq=[self.dataset_image, data_image],
                                           dim=0)
            self.dataset_aux = torch.cat(seq=[self.dataset_aux, data_aux],
                                         dim=0)

            if self.dataset_image.size()[0] > self.dataset_limit:
                self.dataset_image = self.dataset_image.narrow(dimension=0,
                                                               start=self.dataset_image.size()[0] - self.dataset_limit,
                                                               length=self.dataset_limit)
                self.dataset_aux = self.dataset_aux.narrow(dimension=0,
                                                           start=self.dataset_aux.size()[0] - self.dataset_limit,
                                                           length=self.dataset_limit)

    def load_models(self):

        print('Trying load models....')
        try:
            self.netD.load_state_dict(torch.load('{0}/netD.pth'.format(config.modeldir)))
            print('Previous checkpoint for netD founded')
        except Exception, e:
            print('Previous checkpoint for netD unfounded')
        try:
            self.netG.load_state_dict(torch.load('{0}/netG.pth'.format(config.modeldir)))
            print('Previous checkpoint for netC founded')
        except Exception, e:
            print('Previous checkpoint for netC unfounded')

    def save_models(self):
        '''do checkpointing'''
        torch.save(self.netD.state_dict(), '{0}/netD.pth'.format(config.modeldir))
        torch.save(self.netG.state_dict(), '{0}/netG.pth'.format(config.modeldir))

    def save_sample(self,sample,name):

        '''function need for log image'''
        def sample2image(sample):
            if config.gan_nc is 1:
                c = sample / 3.0
                c = torch.unsqueeze(c,1)
                save = torch.cat([c,c,c],1)
            elif config.gan_nc is 3:
                save = []
                for image_i in range(sample.size()[0]/config.gan_nc):
                    save += [torch.unsqueeze(sample.narrow(0,image_i*3,3),0)]
                save = torch.cat(save,0)
            
            # save = save.mul(0.5).add(0.5)
            return save

        number_rows = 4

        '''log real result'''
        sample = sample2image(sample)
        vutils.save_image(sample, ('{0}/'+(name+'<'+config.lable+'>')+'_{1}.png').format(self.experiment, self.iteration_i),number_rows)
        sample = sample.cpu().numpy()
        vis.images( sample,
                    win=(name+'<'+config.lable+'>'),
                    opts=dict(caption=(name+'<'+config.lable+'>')+str(self.iteration_i)))

    def if_dataset_full(self):
        if self.dataset_image.size()[0] >= self.dataset_limit:
            return True
        else:
            return False

    def line(self,x,name):
        x = x.cpu()
        vis.line(   x,
                    win=(name+'<'+config.lable+'>'),
                    opts=dict(title=(name+'<'+config.lable+'>')))
        plt.figure()
        temp, = plt.plot(x.numpy(),alpha=0.5,label=(name+'<'+config.lable+'>'))
        plt.legend(handles=[temp])
        plt.savefig(self.experiment+'/'+(name+'<'+config.lable+'>')+'.jpg')

    def heatmap(self, x, name):
        x = x.permute(1,0)
        x = x.cpu()
        vis.heatmap(    x,
                        win=(name+'<'+config.lable+'>'),
                        opts=dict(title=(name+'<'+config.lable+'>')))
