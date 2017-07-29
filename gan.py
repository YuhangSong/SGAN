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
        self.recorder_loss_g_from_d = torch.FloatTensor(0)
        self.recorder_loss_g_from_c = torch.FloatTensor(0)
        self.recorder_loss_g_from_d_maped = torch.FloatTensor(0)
        self.recorder_loss_g_from_c_maped = torch.FloatTensor(0)
        self.recorder_loss_g_from_mse_maped = torch.FloatTensor(0)
        self.recorder_loss_g = torch.FloatTensor(0)
        self.recorder_loss_d_fake = torch.FloatTensor(0)
        self.recorder_loss_d_real = torch.FloatTensor(0)

        self.indexs_selector = torch.LongTensor(self.batchSize)

        self.mse_loss_model = torch.nn.MSELoss()

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
            self.mse_loss_model = self.mse_loss_model.cuda()

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

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

                ################# load a trained batch #####################
                # generate indexs
                indexs = self.indexs_selector.random_(0,self.dataset_image.size()[0]).cuda()

                # indexing image
                image = self.dataset_image.index_select(0,indexs).cuda()
                state_prediction_gt = torch.cat([image.narrow(1,0,1),image.narrow(1,1,1),image.narrow(1,2,1),image.narrow(1,3,1)],2)
                # image part to
                self.state_prediction_gt = torch.squeeze(state_prediction_gt,1)
                self.state = self.state_prediction_gt.narrow(1,0*self.nc,3*self.nc)
                self.prediction_gt = self.state_prediction_gt.narrow(1,3*self.nc,1*self.nc)
                # indexing aux
                self.aux = self.dataset_aux.index_select(0,indexs).cuda()
                #####################################################

                ################# process prediction_gt #################
                to_cat = [self.prediction_gt]*3
                prediction_gt_x3 = torch.cat(to_cat,1)

                # feed
                self.inputg_image.resize_as_(prediction_gt_x3).copy_(prediction_gt_x3)
                inputg_image_v = Variable(self.inputg_image, volatile = True)

                self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
                inputg_aux_v = Variable(self.inputg_aux, volatile = True)

                self.noise.resize_(self.batchSize, self.aux_size).normal_(0, 1)
                noise_v = Variable(self.noise, volatile = True)

                # compute encoded
                state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                                input_aux_v=inputg_aux_v,
                                                input_noise_v=noise_v)

                self.prediction_gt = state_prediction_v.narrow(1,0,self.nc)

                #####################################################

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

                self.recorder_loss_d_real = torch.cat([self.recorder_loss_d_real,errD_real.data.cpu()],0)
                #####################################################

                ###################### get fake #####################

                # feed
                self.inputg_image.resize_as_(self.state).copy_(self.state)
                inputg_image_v = Variable(self.inputg_image, volatile = True) # totally freeze

                self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
                inputg_aux_v = Variable(self.inputg_aux, volatile = True) # totally freeze

                self.noise.resize_(self.batchSize, self.aux_size).normal_(0, 1)
                noise_v = Variable(self.noise, volatile = True) # totally freeze

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

                self.recorder_loss_d_fake = torch.cat([self.recorder_loss_d_fake,errD_fake.data.cpu()],0)

                # optmize
                errD = errD_real - errD_fake
                #####################################################

                self.optimizerD.step()

                ############# train C with real & fake ##############
                if config.train_corrector:

                    # clamp parameters to a cube
                    for p in self.netC.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    # reset grandient
                    self.netC.zero_grad()

                    # feed real

                    self.state_prediction_gt_ = torch.cat([self.state_prediction_gt, self.state_prediction], 0)
                    self.inputc_image_2.resize_as_(self.state_prediction_gt_).copy_(self.state_prediction_gt_)
                    inputc_image_v = Variable(self.inputc_image_2)

                    self.aux_gt_ = torch.cat([self.aux, self.aux], 0)
                    self.inputc_aux_2.resize_as_(self.aux_gt_).copy_(self.aux_gt_)
                    inputc_aux_v = Variable(self.inputc_aux_2)

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
            for p in self.netD.parameters():
                p.requires_grad = False

            if config.gan_gloss_c_porpotion > 0.0:
                for p in self.netC.parameters():
                    p.requires_grad = False

            # reset grandient
            self.netG.zero_grad()

            # feed
            self.inputg_image.resize_as_(self.state).copy_(self.state)
            inputg_image_v = Variable(self.inputg_image)

            self.inputg_aux.resize_as_(self.aux).copy_(self.aux)
            inputg_aux_v = Variable(self.inputg_aux)

            self.noise.resize_(self.batchSize, self.aux_size).normal_(0, 1)
            noise_v = Variable(self.noise)

            # predict
            state_prediction_v = self.netG( input_image_v=inputg_image_v,
                                            input_aux_v=inputg_aux_v,
                                            input_noise_v=noise_v)

            prediction_v = state_prediction_v.narrow(1,self.nc*3,self.nc)
            state_v = state_prediction_v.narrow(1,0,self.nc*3)

            loss_mse = self.mse_loss_model(state_v,inputg_image_v)
            loss_mse_maped = loss_mse * 4.0
            self.recorder_loss_g_from_mse_maped = torch.cat([self.recorder_loss_g_from_mse_maped,loss_mse_maped.data.cpu()],0)

            # get state_predictionv, this is a Variable cat 
            state_v_prediction_v = torch.cat([Variable(self.state), prediction_v], 1)

            # feed, this state_predictionv is Variable
            inputd_image_v = state_v_prediction_v

            # feed
            self.inputd_aux.resize_as_(self.aux).copy_(self.aux)
            inputd_aux_v = Variable(self.inputd_aux)

            errG_from_D_v, _ = self.netD(input_image_v=inputd_image_v,
                                         input_aux_v=inputd_aux_v)
            self.recorder_loss_g_from_d = torch.cat([self.recorder_loss_g_from_d,errG_from_D_v.data.cpu()],0)

            if config.gan_gloss_c_porpotion > 0.0:

                # errG_from_D is mapped for integrationg with errG_from_C latter
                errG_from_D_const = errG_from_D_v.data.cpu().numpy()[0] # avoid grandient
                errG_from_D_v_maped = torch.mul(torch.mul(errG_from_D_v,(config.auto_d_c_factor**errG_from_D_const)),(1-config.gan_gloss_c_porpotion)) # maped
                self.recorder_loss_g_from_d_maped = torch.cat([self.recorder_loss_g_from_d_maped,errG_from_D_v_maped.data.cpu()],0) # record

            if config.gan_gloss_c_porpotion > 0.0:

                inputc_image_v = inputd_image_v

                self.inputc_aux.resize_as_(self.aux).copy_(self.aux)
                inputc_aux_v = Variable(self.inputc_aux)

                # compute
                outputc = self.netC(inputc_image_v, inputc_aux_v)

                errG_from_C_v = torch.nn.functional.binary_cross_entropy(outputc,self.ones_v)
                self.recorder_loss_g_from_c = torch.cat([self.recorder_loss_g_from_c,errG_from_C_v.data.cpu()],0)

                # avoid grandient
                errG_from_C_const = errG_from_C_v.data.cpu().numpy()[0]

                errG_from_C_v_maped = torch.mul(torch.mul(errG_from_C_v,(config.auto_d_c_factor**errG_from_C_const)),(config.gan_gloss_c_porpotion))
                self.recorder_loss_g_from_c_maped = torch.cat([self.recorder_loss_g_from_c_maped,errG_from_C_v_maped.data.cpu()],0)

            if config.gan_gloss_c_porpotion > 0.0:
                print(s)
                errG = errG_from_D_v_maped + errG_from_C_v_maped + loss_mse_maped
            else:
                errG = errG_from_D_v + loss_mse_maped

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

            '''log result'''
            print('[iteration_i:%d] Loss_D:%f Loss_G:%f Loss_D_real:%f Loss_D_fake:%f Loss_G_mse:%f'
                % (self.iteration_i,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], loss_mse_maped.data[0]))

            '''log image result and save models'''
            if (time.time()-self.last_save_model_time) > config.gan_worker_com_internal:
                self.save_models()
                self.last_save_model_time = time.time()

            if (time.time()-self.last_save_image_time) > config.gan_save_image_internal:

                # feed
                multiple_one_state = torch.cat([self.state[0:1]]*self.batchSize,0)

                self.inputg_image.resize_as_(multiple_one_state).copy_(multiple_one_state)
                inputg_image_v = Variable(self.inputg_image)

                multiple_one_aux = torch.cat([self.aux[0:1]]*self.batchSize,0)
                self.inputg_aux.resize_as_(multiple_one_aux).copy_(multiple_one_aux)
                inputg_aux_v = Variable(self.inputg_aux) # totally freeze netG

                self.noise.resize_(self.batchSize, self.aux_size).normal_(0, 1)
                noise_v = Variable(self.noise) # totally freeze netG

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
                    self.save_sample(self.state_prediction[0],'fake_'+('%.5f'%(outputc_gt_[self.batchSize].data.cpu().numpy()[0])).replace('.',''))

                '''log'''
                plt.figure()
                line_loss_d_real, = plt.plot(self.recorder_loss_d_real.cpu().numpy(),alpha=0.5,label='loss_d_real')
                line_loss_d_fake, = plt.plot(self.recorder_loss_d_fake.cpu().numpy(),alpha=0.5,label='loss_d_fake')
                plt.legend(handles=[line_loss_d_real, line_loss_d_fake])
                plt.savefig(self.experiment+'/loss_d_rf.jpg')

                plt.figure()
                line_loss_d, = plt.plot(self.recorder_loss_g_from_d.cpu().numpy(),alpha=0.5,label='loss_d')
                line_loss_c, = plt.plot(self.recorder_loss_g_from_c.cpu().numpy(),alpha=0.5,label='loss_c')
                plt.legend(handles=[line_loss_d, line_loss_c])
                plt.savefig(self.experiment+'/loss_d_c.jpg')

                plt.figure()
                line_loss_g_from_d_maped, = plt.plot(self.recorder_loss_g_from_d_maped.cpu().numpy(),alpha=0.5,label='loss_g_from_d_maped')
                line_loss_g_from_c_maped, = plt.plot(self.recorder_loss_g_from_c_maped.cpu().numpy(),alpha=0.5,label='loss_g_from_c_maped')
                line_loss_g_from_mse_maped, = plt.plot(self.recorder_loss_g_from_mse_maped.cpu().numpy(),alpha=0.5,label='loss_g_from_mse_maped')
                line_loss_g, = plt.plot(self.recorder_loss_g.cpu().numpy(),alpha=0.5,label='loss_g')
                plt.legend(handles=[line_loss_g_from_d_maped, line_loss_g_from_c_maped, line_loss_g_from_mse_maped, line_loss_g])
                plt.savefig(self.experiment+'/loss_g.jpg')

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
        # torch.save(self.netC.state_dict(), '{0}/{1}/netC.pth'.format(self.experiment,config.gan_model_name_))
        torch.save(self.netG.state_dict(), '{0}/{1}/netG_Cv.pth'.format(self.experiment,config.gan_model_name_))

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
                    save += [torch.unsqueeze(sample.narrow(0,image_i*3,3),0)]
                save = torch.cat(save,0)
            
            # save = save.mul(0.5).add(0.5)
            return save

        number_rows = 4

        '''log real result'''
        vutils.save_image(sample2image(sample), ('{0}/'+name+'_{1}.png').format(self.experiment, self.iteration_i),number_rows)

    def if_dataset_full(self):
        if self.dataset_image.size()[0] >= self.dataset_limit:
            return True
        else:
            return False
