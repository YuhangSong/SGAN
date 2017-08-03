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
        self.netG_Cv = dcgan.DCGAN_G_Cv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netG_DeCv = dcgan.DCGAN_G_DeCv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netD = dcgan.DCGAN_D(self.imageSize, self.nz, self.nc, self.ndf, self.ngpu, self.n_extra_layers)
        self.netC = dcgan.DCGAN_C(self.imageSize, self.nz, self.nc, self.ndf, self.ngpu, self.n_extra_layers)

        '''init models'''
        self.netG_Cv.apply(weights_init)
        self.netG_DeCv.apply(weights_init)
        self.netD.apply(weights_init)
        self.netC.apply(weights_init)

        self.load_models()

        '''print the models'''
        print(self.netD)
        print(self.netC)
        print(self.netG_Cv)
        print(self.netG_DeCv)

        '''feed interface initialize'''
        # input of d
        self.inputd_image = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputd_aux = torch.FloatTensor(self.batchSize, self.aux_size)
        # input of c
        self.inputc_image = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputc_aux = torch.FloatTensor(self.batchSize, self.aux_size)
        self.inputc_image_2 = torch.FloatTensor(self.batchSize*2, 4, self.imageSize, self.imageSize)
        self.inputc_aux_2 = torch.FloatTensor(self.batchSize*2, self.aux_size)
        # input of g
        self.inputg_image = torch.FloatTensor(self.batchSize, 3, self.imageSize, self.imageSize)
        self.inputg_aux = torch.FloatTensor(self.batchSize, self.aux_size)
        # noise
        self.noise = torch.FloatTensor(self.batchSize, self.aux_size, 1, 1)
        self.fixed_noise = torch.FloatTensor(self.batchSize, self.aux_size, 1, 1).normal_(0, 1)
        # constent
        self.one = torch.FloatTensor([1])
        self.one_v = Variable(self.one)
        self.zero = torch.FloatTensor([0])
        self.mone = self.one * -1
        self.ones_zeros = torch.FloatTensor(np.concatenate((np.ones((self.batchSize)),np.zeros((self.batchSize))),0))
        self.ones_zeros_v = Variable(self.ones_zeros)
        self.ones = torch.FloatTensor(np.ones((self.batchSize)))
        self.ones_v = Variable(self.ones)

        '''dataset intialize'''
        self.dataset_image = torch.FloatTensor(np.zeros((1, 4, self.nc, self.imageSize, self.imageSize)))
        self.dataset_aux = torch.FloatTensor(np.zeros((1, self.aux_size)))

        '''recorders'''
        self.recorder_loss_g_from_d = torch.FloatTensor(0)
        self.recorder_loss_g_from_c = torch.FloatTensor(0)
        self.recorder_loss_g_from_d_maped = torch.FloatTensor(0)
        self.recorder_loss_g_from_c_maped = torch.FloatTensor(0)
        self.recorder_loss_g = torch.FloatTensor(0)

        self.indexs_selector = torch.LongTensor(self.batchSize)

        '''convert tesors to cuda type'''
        if self.cuda:

            self.netD.cuda()
            self.netC.cuda()
            self.netG_Cv.cuda()
            self.netG_DeCv.cuda()

            self.inputd_image = self.inputd_image.cuda()
            self.inputd_aux = self.inputd_aux.cuda()
            self.inputc_image = self.inputc_image.cuda()
            self.inputc_aux = self.inputc_aux.cuda()
            self.inputc_image_2 = self.inputc_image_2.cuda()
            self.inputc_aux_2 = self.inputc_aux_2.cuda()
            self.inputg_image = self.inputg_image.cuda()
            self.inputg_aux = self.inputg_aux.cuda()

            self.dataset_image = self.dataset_image.cuda()
            self.dataset_aux = self.dataset_aux.cuda()

            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.zero = self.zero.cuda()
            self.ones_zeros = self.ones_zeros.cuda()
            self.ones_zeros_v = self.ones_zeros_v.cuda()
            self.ones = self.ones.cuda()
            self.ones_v = self.ones_v.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()

            self.recorder_loss_g_from_d = self.recorder_loss_g_from_d.cuda()
            self.recorder_loss_g_from_c = self.recorder_loss_g_from_c.cuda()
            self.recorder_loss_g_from_d_maped = self.recorder_loss_g_from_d.cuda()
            self.recorder_loss_g_from_c_maped = self.recorder_loss_g_from_c.cuda()
            self.recorder_loss_g = self.recorder_loss_g.cuda()

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerC = optim.RMSprop(self.netC.parameters(), lr = self.lrC)
        self.optimizerG_Cv = optim.RMSprop(self.netG_Cv.parameters(), lr = self.lrG)
        self.optimizerG_DeCv = optim.RMSprop(self.netG_DeCv.parameters(), lr = self.lrG)

        self.iteration_i = 0
        self.last_save_model_time = 0
        self.last_save_image_time = 0

    def train(self):
        """
        train one iteraction
        """

        if self.dataset_image.size()[0] >= self.batchSize:

            '''only train when have enough dataset'''
            print('Train on dataset: '+str(int(self.dataset_image.size()[0])))

            ######################################################################
            ########################### Update D network #########################
            ######################################################################

            '''
                when train D network, paramters of D network in trained,
                reset requires_grad of D network to true.
                (they are set to False below in netG update)
            '''
            if config.gan_gloss_c_porpotion < 1.0:
                for p in self.netD.parameters():
                    p.requires_grad = True
                
            if config.gan_gloss_c_porpotion > 0.0:
                for p in self.netC.parameters():
                    p.requires_grad = True

            '''
                train the discriminator DCiters times
                DCiters is set to 100 only on the first 25 generator iterations or
                very sporadically (once every 500 generator iterations).
                This helps to start with the critic at optimum even in the first iterations.
                There shouldn't be a major difference in performance, but it can help,
                especially when visualizing learning curves (since otherwise you'd see the
                loss going up until the critic is properly trained).
                This is also why the first 25 iterations take significantly longer than
                the rest of the training as well.
            '''
            if self.iteration_i < 25 or self.iteration_i % 500 == 0:
                DCiters = 100
            else:
                DCiters = self.DCiters_

            '''
                start interation training of D network
                D network is trained for sevrel steps when 
                G network is trained for one time
            '''
            j = 0
            # DCiters = 2 # debuging
            while j < DCiters:

                j += 1

                if config.gan_gloss_c_porpotion < 1.0:
                    # clamp parameters to a cube
                    for p in self.netD.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                if config.gan_gloss_c_porpotion > 0.0:
                    # clamp parameters to a cube
                    for p in self.netC.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                ################# load a trained batch #####################

                # generate indexs
                indexs = self.indexs_selector.random_(0,self.dataset_image.size()[0]).cuda()

                # indexing image
                image = self.dataset_image.index_select(0,indexs)
                state_prediction_gt = torch.cat([image.narrow(1,0,1),image.narrow(1,1,1),image.narrow(1,2,1),image.narrow(1,3,1)],2)
                # image part to
                self.state_prediction_gt = torch.squeeze(state_prediction_gt,1)
                self.state = self.state_prediction_gt.narrow(1,0*self.nc,3*self.nc)
                self.prediction_gt = self.state_prediction_gt.narrow(1,3*self.nc,1*self.nc)
                
                # indexing aux
                self.aux = self.dataset_aux.index_select(0,indexs)

                ###################### get fake #####################

                # feed
                self.inputg_image.resize_as_(self.state).copy_(self.state)
                inputg_image_v = Variable(self.inputg_image, volatile = True) # totally freeze netG

                # compute encoded
                encoded_v = self.netG_Cv(inputg_image_v)

                # feed aux
                self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
                inputg_aux_v = Variable(self.inputg_aux, volatile = True) # totally freeze netG
                inputg_aux_v = torch.unsqueeze(inputg_aux_v,2)
                inputg_aux_v = torch.unsqueeze(inputg_aux_v,3)

                # feed noise
                self.noise.resize_(self.batchSize, self.aux_size, 1, 1).normal_(0, 1)
                noise_v = Variable(self.noise, volatile = True) # totally freeze netG

                # concate encoded_v, noise_v, action
                concated = [encoded_v,inputg_aux_v,noise_v]
                encoded_v_noise_v_action_v = torch.cat(concated,1)

                # print(encoded_v_noise_v_actionv.size()) # (64L, 512L, 1L, 1L)

                # predict
                prediction_v = self.netG_DeCv(encoded_v_noise_v_action_v)
                prediction = prediction_v.data

                # get state_prediction
                self.state_prediction = torch.cat([self.state, prediction], 1)

                #####################################################
                    

                if config.gan_gloss_c_porpotion < 1.0:

                    # reset grandient
                    self.netD.zero_grad()

                    ################# train D with real #################

                    # feed
                    self.inputd_image.resize_as_(self.state_prediction_gt).copy_(self.state_prediction_gt)
                    inputd_image_v = Variable(self.inputd_image)
                    self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
                    inputd_aux_v = Variable(self.inputd_aux)
                    inputd_aux_v = torch.unsqueeze(inputd_aux_v,2)
                    inputd_aux_v = torch.unsqueeze(inputd_aux_v,3)

                    # compute
                    errD_real, outputD_real = self.netD(inputd_image_v, inputd_aux_v)
                    errD_real.backward(self.one)

                    #####################################################

                    ################# train D with fake #################

                    # feed
                    self.inputd_image.resize_as_(self.state_prediction).copy_(self.state_prediction)
                    inputd_image_v = Variable(self.inputd_image)
                    self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
                    inputd_aux_v = Variable(self.inputd_aux)
                    inputd_aux_v = torch.unsqueeze(inputd_aux_v,2)
                    inputd_aux_v = torch.unsqueeze(inputd_aux_v,3)

                    # compute
                    errD_fake, outputD_fake = self.netD(inputd_image_v, inputd_aux_v)
                    errD_fake.backward(self.mone)

                    #####################################################

                    # optmize
                    errD = errD_real - errD_fake

                    self.optimizerD.step()

                if config.gan_gloss_c_porpotion > 0.0:

                    # reset grandient
                    self.netC.zero_grad()

                    ############# train C with real & fake ##############
                                    
                    # feed real
                    self.state_prediction_gt_ = torch.cat([self.state_prediction_gt, self.state_prediction], 0)
                    self.inputc_image_2.resize_as_(self.state_prediction_gt_).copy_(self.state_prediction_gt_)
                    inputc_image_v = Variable(self.inputc_image_2)

                    self.aux_gt_ = torch.cat([self.aux, self.aux], 0)
                    self.inputc_aux_2.resize_as_(self.aux_gt_).copy_(self.aux_gt_)
                    inputc_aux_v = Variable(self.inputc_aux_2)
                    inputc_aux_v = torch.unsqueeze(inputc_aux_v,2)
                    inputc_aux_v = torch.unsqueeze(inputc_aux_v,3)

                    # compute
                    outputc = self.netC(inputc_image_v, inputc_aux_v)

                    outputc_gt_ = outputc
     
                    lossc = torch.nn.functional.binary_cross_entropy(outputc,self.ones_zeros_v)

                    lossc.backward()

                    self.optimizerC.step()

                #####################################################

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
            if config.gan_gloss_c_porpotion < 1.0:
                for p in self.netD.parameters():
                    p.requires_grad = False

            if config.gan_gloss_c_porpotion > 0.0:
                for p in self.netC.parameters():
                    p.requires_grad = False

            # reset grandient
            self.netG_Cv.zero_grad()
            self.netG_DeCv.zero_grad()

            # feed
            self.inputg_image.resize_as_(self.state).copy_(self.state)
            inputg_image_v = Variable(self.inputg_image)

            # compute encoded_v
            encoded_v = self.netG_Cv(inputg_image_v)

            # feed aux
            self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
            inputg_aux_v = Variable(self.inputg_aux)
            inputg_aux_v = torch.unsqueeze(inputg_aux_v,2)
            inputg_aux_v = torch.unsqueeze(inputg_aux_v,3)

            # feed noise
            self.noise.resize_(self.batchSize, self.aux_size, 1, 1).normal_(0, 1)
            noise_v = Variable(self.noise)

            # concate encoded_v, noise_v, action
            concated = [encoded_v,inputg_aux_v,noise_v]
            encoded_v_noise_v_action_v = torch.cat(concated,1)

            # predict
            prediction_v = self.netG_DeCv(encoded_v_noise_v_action_v)

            # get state_predictionv, this is a Variable cat 
            state_v_prediction_v = torch.cat([Variable(self.state), prediction_v], 1)

            if config.gan_gloss_c_porpotion < 1.0:

                # feed, this state_predictionv is Variable
                inputd_image_v = state_v_prediction_v

                # feed
                self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
                inputd_aux_v = Variable(self.inputd_aux)
                inputd_aux_v = torch.unsqueeze(inputd_aux_v,2)
                inputd_aux_v = torch.unsqueeze(inputd_aux_v,3)

                # compute errG_from_D_v
                '''it is from -4 to 4 about
                if the generated one is more real, it would be smaller'''
                errG_from_D_v, _ = self.netD(inputd_image_v, inputd_aux_v)
                self.recorder_loss_g_from_d = torch.cat([self.recorder_loss_g_from_d,errG_from_D_v.data],0)

                # avoid grandient
                errG_from_D_const = errG_from_D_v.data.cpu().numpy()[0]

                errG_from_D_v_maped = torch.mul(torch.mul(errG_from_D_v,(config.auto_d_c_factor**errG_from_D_const)),(1-config.gan_gloss_c_porpotion))
                self.recorder_loss_g_from_d_maped = torch.cat([self.recorder_loss_g_from_d_maped,errG_from_D_v_maped.data],0)

            if config.gan_gloss_c_porpotion > 0.0:

                inputc_image_v = state_v_prediction_v

                self.inputc_aux.resize_as_(self.aux).copy_(self.aux)
                inputc_aux_v = Variable(self.inputc_aux)
                inputc_aux_v = torch.unsqueeze(inputc_aux_v,2)
                inputc_aux_v = torch.unsqueeze(inputc_aux_v,3)

                # compute
                outputc = self.netC(inputc_image_v, inputc_aux_v)

                errG_from_C_v = torch.nn.functional.binary_cross_entropy(outputc,self.ones_v)
                self.recorder_loss_g_from_c = torch.cat([self.recorder_loss_g_from_c,errG_from_C_v.data],0)

                # avoid grandient
                errG_from_C_const = errG_from_C_v.data.cpu().numpy()[0]

                errG_from_C_v_maped = torch.mul(torch.mul(errG_from_C_v,(config.auto_d_c_factor**errG_from_C_const)),(config.gan_gloss_c_porpotion))
                self.recorder_loss_g_from_c_maped = torch.cat([self.recorder_loss_g_from_c_maped,errG_from_C_v_maped.data],0)

            if (config.gan_gloss_c_porpotion > 0.0) and (config.gan_gloss_c_porpotion < 1.0):
                errG = errG_from_D_v_maped + errG_from_C_v_maped
            elif config.gan_gloss_c_porpotion == 0.0:
                errG = errG_from_D_v
            elif config.gan_gloss_c_porpotion == 1.0:
                errG = errG_from_C_v

            self.recorder_loss_g = torch.cat([self.recorder_loss_g,errG.data],0)

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
            print('[iteration_i:%d] Loss_G: %.4f'
                % (self.iteration_i,
                errG.data[0]))

            '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image_time) > config.gan_save_image_internal:

                # feed
                multiple_one_state = torch.cat([self.state[0:1]]*self.batchSize,0)
                # print(s)
                self.inputg_image.resize_as_(multiple_one_state).copy_(multiple_one_state)
                inputg_image_v = Variable(self.inputg_image)

                # compute encoded_v
                encoded_v = self.netG_Cv(inputg_image_v)

                # feed aux
                multiple_one_aux = torch.cat([self.aux[0:1]]*self.batchSize,0)
                self.inputg_aux.resize_as_(multiple_one_aux).copy_(multiple_one_aux)
                inputg_aux_v = Variable(self.inputg_aux)
                inputg_aux_v = torch.unsqueeze(inputg_aux_v,2)
                inputg_aux_v = torch.unsqueeze(inputg_aux_v,3)

                # feed noise
                self.noise.resize_(self.batchSize, self.aux_size, 1, 1).normal_(0, 1)
                noise_v = Variable(self.noise)

                # concate encoded_v, noise_v, action
                concated = [encoded_v,inputg_aux_v,noise_v]
                encoded_v_noise_v_action_v = torch.cat(concated,1)

                # predict
                prediction_v = self.netG_DeCv(encoded_v_noise_v_action_v)

                # get state_predictionv, this is a Variable cat 
                state_v_prediction_v = torch.cat([Variable(multiple_one_state), prediction_v], 1)

                state_prediction = state_v_prediction_v.data

                state_prediction_gt_ = torch.cat([self.state_prediction_gt[0:1],state_prediction],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'distri')

                state_prediction_mean = torch.sum(state_prediction,0)/(state_prediction.size()[0])
                state_prediction_gt_ = torch.cat([self.state_prediction_gt[0:1],state_prediction_mean],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'mean')

                '''log'''
                if config.gan_gloss_c_porpotion < 1.0:
                    self.line(self.recorder_loss_g_from_d,'self.recorder_loss_g_from_d')
                if config.gan_gloss_c_porpotion > 0.0:
                    self.line(self.recorder_loss_g_from_c,'self.recorder_loss_g_from_c')

                if config.gan_gloss_c_porpotion < 1.0:
                    self.line(self.recorder_loss_g_from_d_maped,'self.recorder_loss_g_from_d_maped')
                if config.gan_gloss_c_porpotion > 0.0:
                    self.line(self.recorder_loss_g_from_c_maped,'self.recorder_loss_g_from_c_maped')
                
                self.line(self.recorder_loss_g,'self.recorder_loss_g')

                self.last_save_image_time = time.time()

            self.iteration_i += 1

            ######################################################################
            ######################### End One in Iteration  ######################
            ######################################################################

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

        data = torch.FloatTensor(data).cuda()

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
        '''do auto checkpoint'''
        print('Trying load models')
        try:
            self.netD.load_state_dict(torch.load('{0}/{1}/netD.pth'.format(self.experiment,config.gan_model_name_)))
            print('Previous checkpoint for netD founded')
        except Exception, e:
            print('Previous checkpoint for netD unfounded')
        try:
            self.netC.load_state_dict(torch.load('{0}/{1}/netC.pth'.format(self.experiment,config.gan_model_name_)))
            print('Previous checkpoint for netC founded')
        except Exception, e:
            print('Previous checkpoint for netC unfounded')
        try:
            self.netG_Cv.load_state_dict(torch.load('{0}/{1}/netG_Cv.pth'.format(self.experiment,config.gan_model_name_)))
            print('Previous checkpoint for netG_Cv founded')
        except Exception, e:
            print('Previous checkpoint for netG_Cv unfounded')
        try:
            self.netG_DeCv.load_state_dict(torch.load('{0}/{1}/netG_DeCv.pth'.format(self.experiment,config.gan_model_name_)))
            print('Previous checkpoint for netG_DeCv founded')
        except Exception, e:
            print('Previous checkpoint for netG_DeCv unfounded')

    def save_models(self):
        '''do checkpointing'''
        torch.save(self.netD.state_dict(), '{0}/{1}/netD.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netC.state_dict(), '{0}/{1}/netC.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netG_Cv.state_dict(), '{0}/{1}/netG_Cv.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netG_DeCv.state_dict(), '{0}/{1}/netG_DeCv.pth'.format(self.experiment,config.gan_model_name_))

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
        sample=sample2image(sample).cpu().numpy()
        vis.images( sample,
                    win=(name+'_'+config.lable),
                    opts=dict(caption=(name+'_'+config.lable)+str(self.iteration_i)))
    def if_dataset_full(self):
        if self.dataset_image.size()[0] >= self.dataset_limit:
            return True
        else:
            return False

    def line(self,x,name):
        vis.line(   x.cpu(),
                    win=(name+'_'+config.lable),
                    opts=dict(title=(name+'_'+config.lable)))