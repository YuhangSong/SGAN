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
        self.lrG = 0.00005
        self.batchSize = config.gan_batchsize
        self.Diters_ = 5
        self.clamp_lower = -0.01
        self.clamp_upper = 0.01
        self.experiment = config.logdir
        self.dataset_limit = 500

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
        self.netG_Cv = dcgan.DCGAN_G_Cv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netG_DeCv = dcgan.DCGAN_G_DeCv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netD = dcgan.DCGAN_D(self.imageSize, self.nz, self.nc, self.ndf, self.ngpu, self.n_extra_layers)

        '''init models'''
        self.netG_Cv.apply(weights_init)
        self.netG_DeCv.apply(weights_init)
        self.netD.apply(weights_init)

        self.load_models()

        '''print the models'''
        print(self.netG_Cv)
        print(self.netG_DeCv)
        print(self.netD)

        self.inputd = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputd_real_part = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputg = torch.FloatTensor(self.batchSize, 3, self.imageSize, self.imageSize)
        self.inputg_action = torch.FloatTensor(self.batchSize, 1, 1, 1)
        self.noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1)
        self.fixed_noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1).normal_(0, 1)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        '''dataset intialize'''
        self.dataset        = torch.FloatTensor(np.zeros((1, 4, self.nc, self.imageSize, self.imageSize)))
        self.dataset_action = torch.FloatTensor(np.zeros((1, 1, 1, 1)))
        self.dataset_sampler_indexs = torch.LongTensor(self.batchSize)

        '''convert tesors to cuda type'''
        if self.cuda:
            self.netD.cuda()
            self.netG_Cv.cuda()
            self.netG_DeCv.cuda()
            self.inputd = self.inputd.cuda()
            self.inputg = self.inputg.cuda()
            self.inputg_action = self.inputg_action.cuda()
            self.one, self.mone = self.one.cuda(), self.mone.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()
            # self.dataset = self.dataset.cuda()
            # self.dataset_sampler_indexs = self.dataset_sampler_indexs.cuda()

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerG_Cv = optim.RMSprop(self.netG_Cv.parameters(), lr = self.lrG)
        self.optimizerG_DeCv = optim.RMSprop(self.netG_DeCv.parameters(), lr = self.lrG)

        self.iteration_i = 0
        self.last_save_model_time = 0
        self.last_save_image = 0

    def train(self):
        """
        train one iteraction
        """

        if self.dataset.size()[0] >= self.batchSize:

            '''only train when have enough dataset'''
            print('Train on dataset: '+str(int(self.dataset.size()[0])))

            ######################################################################
            ########################### Update D network #########################
            ######################################################################

            '''
                when train D network, paramters of D network in trained,
                reset requires_grad of D network to true.
                (they are set to False below in netG update)
            '''
            for p in self.netD.parameters():
                p.requires_grad = True

            '''
                train the discriminator Diters times
                Diters is set to 100 only on the first 25 generator iterations or
                very sporadically (once every 500 generator iterations).
                This helps to start with the critic at optimum even in the first iterations.
                There shouldn't be a major difference in performance, but it can help,
                especially when visualizing learning curves (since otherwise you'd see the
                loss going up until the critic is properly trained).
                This is also why the first 25 iterations take significantly longer than
                the rest of the training as well.
            '''
            if self.iteration_i < 25 or self.iteration_i % 500 == 0:
                Diters = 100
            else:
                Diters = self.Diters_

            '''
                start interation training of D network
                D network is trained for sevrel steps when 
                G network is trained for one time
            '''
            j = 0
            while j < Diters:
                j += 1

                # clamp parameters to a cube
                for p in self.netD.parameters():
                    p.data.clamp_(self.clamp_lower, self.clamp_upper)

                ######## train D network with real #######

                ## random sample from dataset ##
                index = self.dataset_sampler_indexs.random_(0,self.dataset.size()[0])
                raw = torch.index_select(self.dataset,0,index)
                action = torch.index_select(self.dataset_action,0,index)
                image = []
                for image_i in range(4):
                    image += [raw.narrow(1,image_i,1)]
                state_prediction_gt = torch.cat(image,2)
                state_prediction_gt = torch.squeeze(state_prediction_gt,1)
                if self.cuda:
                    state_prediction_gt = state_prediction_gt.cuda()
                    action = action.cuda()
                state = state_prediction_gt.narrow(1,0*self.nc,3*self.nc)
                prediction_gt = state_prediction_gt.narrow(1,3*self.nc,1*self.nc)

                ######### train D with real ########

                # reset grandient
                self.netD.zero_grad()

                # feed
                self.inputd.resize_as_(state_prediction_gt).copy_(state_prediction_gt)
                inputdv = Variable(self.inputd)

                # compute
                errD_real, outputD_real = self.netD(inputdv)
                errD_real.backward(self.one)

                ########### get fake #############

                # feed
                self.inputg.resize_as_(state).copy_(state)
                inputgv = Variable(self.inputg, volatile = True) # totally freeze netG

                # compute encoded
                encodedv = self.netG_Cv(inputgv)

                # compute noise
                self.noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
                noisev = Variable(self.noise, volatile = True) # totally freeze netG

                # feed
                self.inputg_action.resize_as_(action).copy_(action)
                inputg_actionv = Variable(self.inputg_action, volatile = True) # totally freeze netG

                # concate encodedv, noisev, action
                encodedv_noisev_actionv = torch.cat([encodedv,noisev,inputg_actionv],1)

                # print(encodedv_noisev_actionv.size()) # (64L, 512L, 1L, 1L)

                # predict
                prediction = self.netG_DeCv(encodedv_noisev_actionv)
                prediction = prediction.data
                
                ############ train D with fake ###########

                # get state_prediction
                state_prediction = torch.cat([state, prediction], 1)

                # feed
                self.inputd.resize_as_(state_prediction).copy_(state_prediction)
                inputdv = Variable(self.inputd)

                # compute
                errD_fake, outputD_fake = self.netD(inputdv)
                errD_fake.backward(self.mone)

                # optmize
                errD = errD_real - errD_fake
                self.optimizerD.step()

            ######################################################################
            ####################### End of Update D network ######################
            ######################################################################

            ######################################################################
            ########################## Update G network ##########################
            ######################################################################

            '''
                when train G networks, paramters in p network is freezed
                to avoid computation on grad
                this is reset to true when training D network
            '''
            for p in self.netD.parameters():
                p.requires_grad = False

            # reset grandient
            self.netG_Cv.zero_grad()
            self.netG_DeCv.zero_grad()

            # feed
            self.inputg.resize_as_(state).copy_(state)
            inputgv = Variable(self.inputg)

            # compute encodedv
            encodedv = self.netG_Cv(inputgv)

            # compute noisev
            self.noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
            noisev = Variable(self.noise)

            # feed
            self.inputg_action.resize_as_(action).copy_(action)
            inputg_actionv = Variable(self.inputg_action)

            # concate encodedv and noisev
            encodedv_noisev_actionv = torch.cat([encodedv,noisev,inputg_actionv],1)

            # predict
            prediction = self.netG_DeCv(encodedv_noisev_actionv)

            # get state_predictionv, this is a Variable cat 
            statev_predictionv = torch.cat([Variable(state), prediction], 1)

            # feed, this state_predictionv is Variable
            inputdv = statev_predictionv

            # compute
            errG, _ = self.netD(inputdv)
            errG.backward(self.one)

            # optmize
            self.optimizerG_Cv.step()
            self.optimizerG_DeCv.step()

            ######################################################################
            ###################### End of Update G network #######################
            ######################################################################


            ######################################################################
            ########################### One Iteration ### ########################
            ######################################################################

            '''log result'''
            print('[iteration_i:%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (self.iteration_i,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

            '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image) > config.gan_save_image_internal:
                self.save_sample(state_prediction_gt[0],'real')
                self.save_sample(state_prediction[0],'fake')
                self.last_save_image = time.time()

            self.iteration_i += 1
            ######################################################################
            ######################### End One in Iteration  ######################
            ######################################################################

        else:

            '''dataset not enough'''
            print('Dataset not enough: '+str(int(self.dataset.size()[0])))

            time.sleep(config.gan_worker_com_internal)

    def push_data(self, data):
        """
        push data to dataset which is a torch tensor
        """

        data = torch.FloatTensor(data)

        self.dataset        = torch.cat(seq=[self.dataset,        data[:,0:4,:,:,:]],
                                        dim=0)
        self.dataset_action = torch.cat(seq=[self.dataset_action, np.squeeze(data[:,4:5,0:1,0:1,0:1],4)],
                                        dim=0)

        if self.dataset.size()[0] > self.dataset_limit:
            self.dataset = self.dataset.narrow(dimension=0,
                                               start=self.dataset.size()[0]-self.dataset_limit,
                                               length=self.dataset_limit)

    def load_models(self):
        '''do auto checkpoint'''
        try:
            self.netG_Cv.load_state_dict(torch.load(config.modeldir+'netG_Cv.pth'))
            print('Previous checkpoint for netG_Cv founded')
        except Exception, e:
            print('Previous checkpoint for netG_Cv unfounded')
        try:
            self.netG_DeCv.load_state_dict(torch.load(config.modeldir+'netG_DeCv.pth'))
            print('Previous checkpoint for netG_DeCv founded')
        except Exception, e:
            print('Previous checkpoint for netG_DeCv unfounded')
        try:
            self.netD.load_state_dict(torch.load(config.modeldir+'netD.pth'))
            print('Previous checkpoint for netD founded')
        except Exception, e:
            print('Previous checkpoint for netD unfounded')

    def save_models(self):
        '''do checkpointing'''
        torch.save(self.netG_Cv.state_dict(), '{0}/{1}/netG_Cv.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netG_DeCv.state_dict(), '{0}/{1}/netG_DeCv.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netD.state_dict(), '{0}/{1}/netD.pth'.format(self.experiment,config.gan_model_name_))

    def save_sample(self,sample,name):

        '''function need for log image'''
        def sample2image(sample):
            if config.gan_nc is 1:
                c = sample / 3.0
                c = torch.unsqueeze(c,1)
                save = torch.cat([c,c,c],1)
            elif config.gan_nc is 3:
                save = []
                for image_i in range(4):
                    save += [torch.unsqueeze(sample.narrow(0,image_i*3,3),0)]
                save = torch.cat(save,0)
            
            # save = save.mul(0.5).add(0.5)
            return save

        '''log real result'''
        vutils.save_image(sample2image(sample), ('{0}/'+name+'_{1}.png').format(self.experiment, self.iteration_i))