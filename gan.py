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
import visdom
vis = visdom.Visdom()
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
        self.nc = config.gan_nc
        self.imageSize = config.gan_size
        self.lrD = 0.00005
        self.lrC = 0.00005
        self.lrG = 0.00005
        self.batchSize = config.gan_batchsize
        self.DCiters_ = config.DCiters_
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
        self.netD = dcgan.DCGAN_D()
        self.netG = dcgan.DCGAN_G()

        '''init models'''
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

        self.load_models()

        '''print the models'''
        print(self.netD)
        print(self.netG)

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
        self.noise_image = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
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
        self.recorder_errG_from_D = torch.FloatTensor([0])
        self.recorder_errG_from_niv = torch.FloatTensor([0])
        self.recorder_errR_from_mse = torch.FloatTensor([0])

        self.recorder_loss_g = torch.FloatTensor([0])
        self.recorder_loss_d_fake = torch.FloatTensor([0])
        self.recorder_loss_d_real = torch.FloatTensor([0])

        self.recorder_iteration = torch.FloatTensor([0])

        self.indexs_selector = torch.LongTensor(self.batchSize)

        self.mse_loss_model_averaged = torch.nn.MSELoss(size_average=True)

        '''convert tesors to cuda type'''
        if self.cuda:

            self.netD.cuda()
            self.netG.cuda()

            self.inputd_image = self.inputd_image.cuda()
            self.inputd_aux = self.inputd_aux.cuda()
            self.inputc_image = self.inputc_image.cuda()
            self.inputc_aux = self.inputc_aux.cuda()
            self.inputc_image_2 = self.inputc_image_2.cuda()
            self.inputc_aux_2 = self.inputc_aux_2.cuda()
            self.inputg_image = self.inputg_image.cuda()
            self.inputg_aux = self.inputg_aux.cuda()

            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.zero = self.zero.cuda()
            self.ones_zeros = self.ones_zeros.cuda()
            self.ones_zeros_v = self.ones_zeros_v.cuda()
            self.ones = self.ones.cuda()
            self.ones_v = self.ones_v.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()
            self.noise_image = self.noise_image.cuda()
            self.mse_loss_model_averaged = self.mse_loss_model_averaged.cuda()

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

        self.iteration_i = 0
        self.last_save_model_time = 0
        self.last_save_image_time = 0

        self.training_ruiner_next_time = True
        self.errD_has_been_big = False

    def get_noise(self,size):
        # return self.noise.resize_as_(size).uniform_(0,1).round()
        self.noise.resize_as_(size).zero_()
        for batch_i in range(self.noise.size()[0]):
            index = np.random.randint(0,self.noise.size()[1])
            self.noise[batch_i][index]=1.0
        return self.noise

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

            self.training_ruiner = self.training_ruiner_next_time

            '''
                when train D network, paramters of D network in trained,
                reset requires_grad of D network to true.
                (they are set to False below in netG update)
            '''
            for p in self.netD.parameters():
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
            if self.training_ruiner:
                DCiters = 0
                self.recorder_loss_d_real = torch.cat([self.recorder_loss_d_real,torch.FloatTensor([0.0])],0)
                self.recorder_loss_d_fake = torch.cat([self.recorder_loss_d_fake,torch.FloatTensor([0.0])],0)
            else:
                if self.iteration_i % 500 == 0:
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

                self.get_a_batch(if_ruin_prediction_gt=True)

                # clamp parameters to a cube
                for p in self.netD.parameters():
                    p.data.clamp_(self.clamp_lower, self.clamp_upper)

                ################# train D with real #################
                # reset grandient
                self.netD.zero_grad()

                # feed
                self.inputd_image.resize_as_(self.state_prediction_gt).copy_(self.state_prediction_gt)
                inputd_image_v = Variable(self.inputd_image)
                self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
                inputd_aux_v = Variable(self.inputd_aux)

                # compute
                errD_real, outputD_real = self.netD(input_image_v=inputd_image_v,
                                                    input_aux_v=inputd_aux_v)
                errD_real.backward(self.one)

                if j < 1.0:
                    self.recorder_loss_d_real = torch.cat([self.recorder_loss_d_real,errD_real.data.cpu()],0)
                #####################################################

                ###################### get fake #####################

                # feed
                self.inputg_image.resize_as_(self.state).copy_(self.state)
                inputg_image_v = Variable(self.inputg_image, volatile = True) # totally freeze

                self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
                inputg_aux_v = Variable(self.inputg_aux, volatile = True) # totally freeze

                noise=self.get_noise(torch.cuda.FloatTensor(self.batchSize, self.aux_size))
                noise_v = Variable(noise, volatile = True) # totally freeze

                # predict
                state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                                input_aux_v=inputg_aux_v,
                                                input_noise_v=noise_v)

                prediction_v = state_prediction_v.narrow(1,self.nc*3,self.nc)
                prediction = prediction_v.data
                #####################################################
                
                ################# train D with fake #################
                # get state_prediction
                self.state_prediction = torch.cat([self.state, prediction], 1)

                # feed
                self.inputd_image.resize_as_(self.state_prediction).copy_(self.state_prediction)
                inputd_image_v = Variable(self.inputd_image)
                self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
                inputd_aux_v = Variable(self.inputd_aux)

                # compute
                errD_fake, outputD_fake = self.netD(input_image_v=inputd_image_v,
                                                    input_aux_v=inputd_aux_v)
                errD_fake.backward(self.mone)

                if j < 1.0:
                    self.recorder_loss_d_fake = torch.cat([self.recorder_loss_d_fake,errD_fake.data.cpu()],0)

                # optmize
                errD = errD_real - errD_fake
                #####################################################

                self.optimizerD.step()

                j += 1

            ######################################################################
            ####################### End of Update D network ######################
            ######################################################################

            ######################################################################
            ########################## Update G network ##########################
            ######################################################################

            self.get_a_batch(if_ruin_prediction_gt=True)

            '''
                when train G networks, paramters in p network is freezed
                to avoid computation on grad
                this is reset to true when training D network
            '''
            for p in self.netD.parameters():
                p.requires_grad = False

            # reset grandient
            self.netG.zero_grad()

            # feed
            self.inputg_image.resize_as_(self.state).copy_(self.state)
            inputg_image_v = Variable(self.inputg_image)

            self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
            inputg_aux_v = Variable(self.inputg_aux)

            noise=self.get_noise(torch.cuda.FloatTensor(self.batchSize, self.aux_size))
            noise_v = Variable(noise)

            # predict
            state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                            input_aux_v=inputg_aux_v,
                                            input_noise_v=noise_v)

            prediction_v = state_prediction_v.narrow(1,self.nc*3,self.nc)
            state_v = state_prediction_v.narrow(1,0,self.nc*3)

            # supervise loss from ruiner
            errR_from_mse_v = self.mse_loss_model_averaged(state_v,inputg_image_v)
            errR_from_mse_numpy = errR_from_mse_v.data.cpu().numpy()
            self.recorder_errR_from_mse = torch.cat([self.recorder_errR_from_mse,errR_from_mse_v.data.cpu()],0)

            # keep it uniform
            errR_from_mse_v = errR_from_mse_v * (config.loss_g_factor / errR_from_mse_v.data.cpu().numpy()[0])

            # get state_predictionv, this is a Variable cat 
            state_v_prediction_v = torch.cat([Variable(self.state), prediction_v], 1)

            # feed, this state_predictionv is Variable
            inputd_image_v = state_v_prediction_v

            # feed
            self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
            inputd_aux_v = Variable(self.inputd_aux)

            _, errG_from_D_seperate_v = self.netD(input_image_v=inputd_image_v,
                                         input_aux_v=inputd_aux_v)

            errG_from_D_seperate_v = errG_from_D_seperate_v.squeeze(1)
            
            total_num_numpy = torch.numel(self.prediction_gt) 
            prediction_gt_v = Variable(self.prediction_gt)
            errG_from_niv_seperate_v = torch.pow(torch.add(prediction_v,-prediction_gt_v),2.0) / total_num_numpy
            errG_from_niv_seperate_v = torch.mean(errG_from_niv_seperate_v,3).squeeze(3)
            errG_from_niv_seperate_v = torch.mean(errG_from_niv_seperate_v,2).squeeze(2)
            errG_from_niv_seperate_v = torch.mean(errG_from_niv_seperate_v,1).squeeze(1)

            ######################## get niv #####################################
            '''feed'''
            self.inputd_image.resize_as_(self.state_prediction_gt).copy_(self.state_prediction_gt)
            inputd_image_v = Variable(self.inputd_image)
            self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
            inputd_aux_v = Variable(self.inputd_aux)

            '''compute'''
            _, niv_v = self.netD(   input_image_v=inputd_image_v,
                                    input_aux_v=inputd_aux_v)

            '''revers, so that bigger niv means heavier weight'''
            '''clip, under zero is not add to niv'''
            niv_numpy_remain = np.clip(-niv_v.data.cpu().numpy(),
                                a_min=0.0,
                                a_max=config.loss_g_factor)
            niv_numpy_remain = np.squeeze(niv_numpy_remain,1)

            '''if some is good enough, do not include to Update'''
            niv_numpy = copy.deepcopy(np.asarray(niv_numpy_remain))
            ever_removed = False
            original_shape = np.shape(niv_numpy)
            iii=0
            loss_G_have_niv = True
            while True:

                if iii > (int(np.shape(niv_numpy)[0])-1):
                    break

                if niv_numpy[iii] <= config.donot_niv_gate:
                    ever_removed = True
                    if iii < 1:
                        if iii > (np.shape(niv_numpy)[0]-2):
                            loss_G_have_niv = False
                            break
                        else:
                            niv_numpy = niv_numpy[iii+1:]
                            errG_from_niv_seperate_v = errG_from_niv_seperate_v.narrow(0,(iii+1),errG_from_niv_seperate_v.size()[0]-(iii+1))

                    elif iii > (np.shape(niv_numpy)[0]-2):
                        niv_numpy = niv_numpy[:iii]
                        errG_from_niv_seperate_v = errG_from_niv_seperate_v.narrow(0,0,iii)
                                                           
                    else:
                        niv_numpy = np.concatenate((niv_numpy[:iii],niv_numpy[iii+1:]),0)
                        errG_from_niv_seperate_v = torch.cat(   [errG_from_niv_seperate_v.narrow(0,0,iii),
                                                                 errG_from_niv_seperate_v.narrow(0,(iii+1),errG_from_niv_seperate_v.size()[0]-(iii+1))],
                                                                 0)

                    continue

                else:
                    iii += 1
                

            if loss_G_have_niv:
                now_shape = np.shape(niv_numpy)
                if ever_removed:
                    print('Some elements in errG_from_niv_seperate_v is removed: '+str(original_shape)+'>>>'+str(now_shape))
                else:
                    print('Full elements in errG_from_niv_seperate_v is used: '+str(now_shape))

                niv_numpy = np.expand_dims(niv_numpy,1)
                errG_from_niv_seperate_v = errG_from_niv_seperate_v.unsqueeze(0)

                errG_from_niv_v = torch.mm(errG_from_niv_seperate_v,Variable(torch.from_numpy(niv_numpy).cuda())).squeeze(1) / now_shape[0]
                self.recorder_errG_from_niv = torch.cat([self.recorder_errG_from_niv,errG_from_niv_v.data.cpu()],0)

                # keep it uniform
                errG_from_niv_v = errG_from_niv_v * (config.loss_g_factor / errG_from_niv_v.data.cpu().numpy()[0] * config.niv_rate)
            else:
                self.recorder_errG_from_niv = torch.cat([self.recorder_errG_from_niv,torch.FloatTensor([0.0])],0)
                print('No errG_from_niv_seperate_v.')


            '''if some is good enough, do not include to Update'''
            niv_numpy = copy.deepcopy(np.asarray(niv_numpy_remain))
            ever_removed = False
            original_shape = np.shape(niv_numpy)
            iii=0
            loss_G_have_D = True
            while True:

                if iii > (int(np.shape(niv_numpy)[0])-1):
                    break

                if niv_numpy[iii] > config.donot_niv_gate:
                    ever_removed = True
                    if iii < 1:
                        if iii > (np.shape(niv_numpy)[0]-2):
                            loss_G_have_D = False
                            break
                        else:
                            niv_numpy = niv_numpy[iii+1:]
                            errG_from_D_seperate_v = errG_from_D_seperate_v.narrow(0,(iii+1),errG_from_D_seperate_v.size()[0]-(iii+1))

                    elif iii > (np.shape(niv_numpy)[0]-2):
                        niv_numpy = niv_numpy[:iii]
                        errG_from_D_seperate_v = errG_from_D_seperate_v.narrow(0,0,iii)
                                                           
                    else:
                        niv_numpy = np.concatenate((niv_numpy[:iii],niv_numpy[iii+1:]),0)
                        errG_from_D_seperate_v = torch.cat(   [errG_from_D_seperate_v.narrow(0,0,iii),
                                                                 errG_from_D_seperate_v.narrow(0,(iii+1),errG_from_D_seperate_v.size()[0]-(iii+1))],
                                                                 0)

                    continue

                else:
                    iii += 1
                

            if loss_G_have_D:
                now_shape = np.shape(niv_numpy)
                if ever_removed:
                    print('Some elements in errG_from_D_seperate_v is removed: '+str(original_shape)+'>>>'+str(now_shape))
                else:
                    print('Full elements in errG_from_D_seperate_v is used: '+str(now_shape))

                errG_from_D_v = errG_from_D_seperate_v.mean(0)
                self.recorder_errG_from_D = torch.cat([self.recorder_errG_from_D,errG_from_D_v.data.cpu()],0)

                # keep it uniform
                errG_from_D_v = errG_from_D_v * (config.loss_g_factor / errG_from_D_v.data.cpu().numpy()[0] * config.niv_rate)
                
            else:
                self.recorder_errG_from_D = torch.cat([self.recorder_errG_from_D,torch.FloatTensor([0.0])],0)
                print('No errG_from_D_seperate_v.')


            if errR_from_mse_numpy[0] > config.ruiner_train_to_mse:
                if self.training_ruiner:
                    errG = errR_from_mse_v
                else:
                    errG = errR_from_mse_v
                    if loss_G_have_niv:
                        errG = errG + errG_from_niv_v
                    if loss_G_have_D:
                        errG = errG + errG_from_D_v

            else:
                self.training_ruiner_next_time = False

                if loss_G_have_niv and loss_G_have_D:
                    errG = errG_from_D_v + errG_from_niv_v
                else:
                    if loss_G_have_D:
                        errG = errG_from_D_v
                    elif loss_G_have_niv:
                        errG = errG_from_niv_v
                    else:
                        print(e)


            self.recorder_loss_g = torch.cat([self.recorder_loss_g,errG.data.cpu()],0)

            errG.backward(self.one)

            # optmize
            self.optimizerG.step()

            ######################################################################
            ###################### End of Update G network #######################
            ######################################################################


            ######################################################################
            ########################### One Iteration ### ########################
            ######################################################################

            self.iteration_i += 1
            self.recorder_iteration = torch.cat([self.recorder_iteration,torch.FloatTensor([self.iteration_i])],0)

            if self.training_ruiner:
                print('[iteration_i:%d] Training ruiner >> Loss_R_mse:%.4f'
                    % (self.iteration_i,errR_from_mse_numpy[0]))
            else:
                print('[iteration_i:%d] Loss_D:%.2f Loss_G:%.2f Loss_D_real:%.2f Loss_D_fake:%.2f Loss_R_mse:%.4f'
                    % (self.iteration_i,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], errR_from_mse_numpy[0]))
                
                if self.errD_has_been_big:
                    if abs(errD.data.cpu().numpy()[0]) < config.bloom_at_errD:
                        if config.bloom_noise_step > 0.0:
                            print('>>>>>>>>>>>>>>>>>>>>> Bloom noise >>>>>>>>>>>>>>>>>>>>>')
                            self.netG.bloom_noise()
                else:
                    print('>>>>>>>>>>>>>>>>>>>>> errD has not been big >>>>>>>>>>>>>>>>>>>>>')
                    if abs(errD.data.cpu().numpy()[0]) > config.loss_g_factor:
                        print('>>>>>>>>>>>>>>>>>>>>> errD has been big >>>>>>>>>>>>>>>>>>>>>')
                        self.errD_has_been_big = True

            '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image_time) > config.gan_save_image_internal:

                save_batch_size = self.batchSize
                # feed
                multiple_one_state = torch.cat([self.state[0:1]]*save_batch_size,0)

                self.inputg_image.resize_as_(multiple_one_state).copy_(multiple_one_state)
                inputg_image_v = Variable(self.inputg_image)

                multiple_one_aux = torch.cat([self.aux[0:1]]*save_batch_size,0)
                self.inputg_aux.resize_as_(multiple_one_aux).copy_(multiple_one_aux)
                inputg_aux_v = Variable(self.inputg_aux) # totally freeze netG

                noise=self.get_noise(torch.cuda.FloatTensor(save_batch_size, self.aux_size))
                noise_v = Variable(noise) # totally freeze netG

                # predict
                state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                                input_aux_v=inputg_aux_v,
                                                input_noise_v=noise_v)

                prediction_v = state_prediction_v.narrow(1,self.nc*3,self.nc)
                state_v = state_prediction_v.narrow(1,0,self.nc*3)

                # get state_predictionv, this is a Variable cat 
                state_v_prediction_v = torch.cat([Variable(multiple_one_state), prediction_v], 1)

                state_prediction = state_v_prediction_v.data

                state_prediction_gt_ = torch.cat([self.state_prediction_gt[0:1],state_prediction],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'distri_f',-1)

                state_prediction_mean = torch.sum(state_prediction,0)/(state_prediction.size()[0])
                state_prediction_gt_mean = torch.sum(self.state_prediction_gt,0)/(self.state_prediction_gt.size()[0])
                state_prediction_gt_ = torch.cat([state_prediction_gt_mean,state_prediction_mean],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'mean_f')

                if config.train_corrector:
                    self.save_sample(self.state_prediction_gt[0],'real_'+('%.5f'%(outputc_gt_[0].data.cpu().numpy()[0])).replace('.',''))
                    self.save_sample(self.state_prediction[0],'fake_'+('%.5f'%(outputc_gt_[save_batch_size].data.cpu().numpy()[0])).replace('.',''))

                '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image_time) > config.gan_save_image_internal:

                save_batch_size = self.batchSize
                # feed
                multiple_one_state = torch.cat([self.state[0:1]]*save_batch_size,0)

                self.inputg_image.resize_as_(multiple_one_state).copy_(multiple_one_state)
                inputg_image_v = Variable(self.inputg_image)

                multiple_one_aux = torch.cat([self.aux[0:1]]*save_batch_size,0)
                self.inputg_aux.resize_as_(multiple_one_aux).copy_(multiple_one_aux)
                inputg_aux_v = Variable(self.inputg_aux) # totally freeze netG

                noise = self.get_noise(torch.cuda.FloatTensor(save_batch_size, self.aux_size))
                noise_v = Variable(noise) # totally freeze netG

                # predict
                state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                                input_aux_v=inputg_aux_v,
                                                input_noise_v=noise_v)

                prediction_v = state_prediction_v.narrow(1,self.nc*3,self.nc)
                state_v = state_prediction_v.narrow(1,0,self.nc*3)

                # get state_predictionv, this is a Variable cat 
                state_v_prediction_v = torch.cat([Variable(multiple_one_state), prediction_v], 1)

                state_prediction = state_v_prediction_v.data

                state_prediction_gt_ = torch.cat([self.state_prediction_gt[0:1],state_prediction],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'distri_f',-1)

                state_prediction_mean = torch.sum(state_prediction,0)/(state_prediction.size()[0])
                state_prediction_gt_mean = torch.sum(self.state_prediction_gt,0)/(self.state_prediction_gt.size()[0])
                state_prediction_gt_ = torch.cat([state_prediction_gt_mean,state_prediction_mean],0)
                state_prediction_gt_channelled = state_prediction_gt_[0]
                for batch_i in range(1,state_prediction_gt_.size()[0]):
                    state_prediction_gt_channelled = torch.cat([state_prediction_gt_channelled,state_prediction_gt_[batch_i]],0)
                self.save_sample(state_prediction_gt_channelled,'mean_f')

                if config.train_corrector:
                    self.save_sample(self.state_prediction_gt[0],'real_'+('%.5f'%(outputc_gt_[0].data.cpu().numpy()[0])).replace('.',''))
                    self.save_sample(self.state_prediction[0],'fake_'+('%.5f'%(outputc_gt_[save_batch_size].data.cpu().numpy()[0])).replace('.',''))

                print(self.recorder_iteration.size())
                print(self.recorder_loss_d_real.size())
                '''log'''
                vis.line(   Y=self.recorder_loss_d_real,
                            X=self.recorder_iteration,
                            win='recorder_loss_d_real',
                            opts=dict(title='recorder_loss_d_real'))

                vis.line(   self.recorder_loss_d_fake,
                            win='recorder_loss_d_fake',
                            opts=dict(title='recorder_loss_d_fake'))

                vis.line(   self.recorder_errG_from_niv,
                            win='recorder_errG_from_niv',
                            opts=dict(title='recorder_errG_from_niv'))

                vis.line(   self.recorder_errG_from_D,
                            win='recorder_errG_from_D',
                            opts=dict(title='recorder_errG_from_D'))  

                vis.line(   Y=self.recorder_errR_from_mse,
                            X=self.recorder_iteration,
                            win='recorder_errR_from_mse',
                            opts=dict(title='recorder_errR_from_mse'))

                vis.line(   self.recorder_loss_g,
                            win='loss_g',
                            opts=dict(title='loss_g'))         

                self.last_save_image_time = time.time()

            ######################################################################
            ######################### End One in Iteration  ######################
            ######################################################################

        else:

            '''dataset not enough'''
            print('Dataset not enough: '+str(int(self.dataset_image.size()[0])))

            time.sleep(config.gan_worker_com_internal)

    def get_a_batch(self,if_ruin_prediction_gt):

        ################# load a trained batch #####################
        # generate indexs
        indexs = self.indexs_selector.random_(0,self.dataset_image.size()[0])

        # indexing image
        image = torch.index_select( self.dataset_image,
                                    dim=0,
                                    index=indexs).cuda()

        state_prediction_gt = torch.cat([torch.index_select(image,1,torch.cuda.LongTensor(range(0,1))),torch.index_select(image,1,torch.cuda.LongTensor(range(1,2))),torch.index_select(image,1,torch.cuda.LongTensor(range(2,3))),torch.index_select(image,1,torch.cuda.LongTensor(range(3,4)))],2)
        # image part to
        self.state_prediction_gt = torch.squeeze(state_prediction_gt,1)
        self.state = torch.index_select(self.state_prediction_gt,1,torch.cuda.LongTensor(range(0*self.nc,3*self.nc)))
        self.prediction_gt = torch.index_select(self.state_prediction_gt,1,torch.cuda.LongTensor(range(3*self.nc,4*self.nc)))
        # indexing aux
        self.aux = torch.index_select(  self.dataset_aux,
                                        dim=0,
                                        index=indexs).cuda()
        #############################################################

        if if_ruin_prediction_gt:

            ################# process prediction_gt #####################
            to_cat = [self.prediction_gt]*3
            prediction_gt_x3 = torch.cat(to_cat,1)

            # feed
            self.inputg_image.resize_as_(prediction_gt_x3).copy_(prediction_gt_x3)
            inputg_image_v = Variable(self.inputg_image, volatile = True)

            self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
            inputg_aux_v = Variable(self.inputg_aux, volatile = True)

            noise = self.get_noise(torch.cuda.FloatTensor(self.batchSize, self.aux_size))
            noise_v = Variable(noise, volatile = True)

            # compute encoded
            state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                            input_aux_v=inputg_aux_v,
                                            input_noise_v=noise_v)
            state_prediction = state_prediction_v.data

            self.prediction_gt = torch.index_select(state_prediction,1,torch.cuda.LongTensor(range(0,self.nc)))
            self.state_prediction_gt = torch.cat([self.state,self.prediction_gt],1)
            ##############################################################

    def push_data(self, data):
        """
        push data to dataset which is a torch tensor
        """

        if np.shape(data)[0] is 0:
            return

        data = torch.FloatTensor(data)

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
            self.netG.load_state_dict(torch.load('{0}/{1}/netG.pth'.format(self.experiment,config.gan_model_name_)))
            print('Previous checkpoint for netG founded')
        except Exception, e:
            print('Previous checkpoint for netG unfounded')

    def save_models(self):
        '''do checkpointing'''
        torch.save(self.netD.state_dict(), '{0}/{1}/netD.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netG.state_dict(), '{0}/{1}/netG.pth'.format(self.experiment,config.gan_model_name_))

    def save_sample(self,sample,name,num=-1):

        '''function need for log image'''
        def sample2image(sample):
            if config.gan_nc is 1:
                c = sample / 3.0
                c = torch.unsqueeze(c,1)
                save = torch.cat([c,c,c],1)
            elif config.gan_nc is 3:
                save = []
                if num < 0:
                    image_to = sample.size()[0]/config.gan_nc
                else:
                    image_to = num
                for image_i in range(image_to):
                    save += [torch.unsqueeze(torch.index_select(sample,0,torch.cuda.LongTensor(range(image_i*3,image_i*3+3))),0)]
                save = torch.cat(save,0)
            
            # save = save.mul(0.5).add(0.5)
            return save

        number_rows = 4

        '''log real result'''
        # vutils.save_image(sample2image(sample), ('{0}/'+name+'_{1}.png').format(self.experiment, self.iteration_i),number_rows)
        sample=sample2image(sample).cpu().numpy()
        vis.images( sample,
                    win=name,
                    opts=dict(caption=name+str(self.iteration_i)))

    def if_dataset_full(self):
        if self.dataset_image.size()[0] >= self.dataset_limit:
            return True
        else:
            return False
