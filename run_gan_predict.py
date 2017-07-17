from __future__ import print_function
import argparse
import random
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
import os
import numpy as np
import copy

import wgan_models.dcgan as dcgan
import wgan_models.mlp as mlp
import support_lib
import config
import subprocess
import time
import gsa_io

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lsun', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../../dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--batchSize', type=int, default=config.gan_batchsize, help='input batch size')
parser.add_argument('--imageSize', type=int, default=config.gan_size, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=config.gan_nz, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=config.gan_ngpu, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=config.logdir, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

subprocess.call(["mkdir", "-p", opt.experiment])

# Where to store samples and models
if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}F)'.format(opt.experiment))

# random seed for
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

'''create netG'''
if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    # in the paper. this is the best implementation
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)

# load checkpoint if needed
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

# load checkpoint if needed
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

inputd = torch.FloatTensor(opt.batchSize, 4, opt.imageSize, opt.imageSize)
inputd_real_part = torch.FloatTensor(opt.batchSize, 4, opt.imageSize, opt.imageSize)
inputg = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
one = torch.FloatTensor([1])
mone = one * -1

'''load dataset'''
dataset = np.load(config.dataset_path+config.dataset_name)['dataset']
dataset_len = np.shape(dataset)[0]
print('dataset loaded, size: ' + str(np.shape(dataset)))
dataset = torch.FloatTensor(dataset)
dataset_sampler_indexs = torch.LongTensor(opt.batchSize).random_(0,dataset_len)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    inputd = inputd.cuda()
    inputg = inputg.cuda()
    one, mone = one.cuda(), mone.cuda()
    # dataset = dataset.cuda()
    # dataset_sampler_indexs = dataset_sampler_indexs.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

iteration_i = 0
dataset_i = 0

while True:

    ######################################################################
    ########################### Update D network #########################
    ######################################################################

    '''
        when train D network, paramters of D network in trained,
        reset requires_grad of D network to true.
        (they are set to False below in netG update)
    '''
    for p in netD.parameters():
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
    if iteration_i < 25 or iteration_i % 500 == 0:
        Diters = 100
    else:
        Diters = opt.Diters

    '''
        start interation training of D network
        D network is trained for sevrel steps when 
        G network is trained for one time
    '''
    j = 0
    while j < Diters:
        j += 1

        # clamp parameters to a cube
        for p in netD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        # next dataset
        dataset_i += 1

        ######## train D network with real #######

        ## random sample from dataset ##
        state_prediction_gt = torch.index_select(dataset,0,dataset_sampler_indexs.random_(0,opt.batchSize))
        if opt.cuda:
            state_prediction_gt = state_prediction_gt.cuda()
        state = state_prediction_gt.narrow(1,0,3)
        prediction_gt = state_prediction_gt.narrow(1,3,1)

        ######### train D with real ########

        # reset grandient
        netD.zero_grad()

        # feed
        inputd.resize_as_(state_prediction_gt).copy_(state_prediction_gt)
        inputdv = Variable(inputd)

        # compute
        errD_real, outputD_real = netD(inputdv)
        errD_real.backward(one)

        ########### get fake #############

        # feed
        inputg.resize_as_(state).copy_(state)
        inputgv = Variable(inputg, volatile = True) # totally freeze netG

        # compute
        prediction = netG(inputgv)
        prediction = prediction.data
        
        ############ train D with fake ###########

        # get state_prediction
        state_prediction = torch.cat([state, prediction], 1)

        # feed
        inputd.resize_as_(state_prediction).copy_(state_prediction)
        inputdv = Variable(inputd)

        # compute
        errD_fake, outputD_fake = netD(inputdv)
        errD_fake.backward(mone)

        # optmize
        errD = errD_real - errD_fake
        optimizerD.step()

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
    for p in netD.parameters():
        p.requires_grad = False

    # reset grandient
    netG.zero_grad()

    # feed
    inputg.resize_as_(state).copy_(state)
    inputgv = Variable(inputg)

    # compute
    prediction = netG(inputgv)

    # get state_predictionv, this is a Variable cat 
    state_predictionv = torch.cat([Variable(state), prediction], 1)

    # feed, this state_predictionv is Variable
    inputdv = state_predictionv

    # compute
    errG, _ = netD(inputdv)
    errG.backward(one)

    # optmize
    optimizerG.step()

    ######################################################################
    ###################### End of Update G network #######################
    ######################################################################


    ######################################################################
    ########################### One Iteration ### ########################
    ######################################################################

    '''log result'''
    print('[iteration_i:%d][dataset_i:%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
        % (iteration_i, iteration_i,
        errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

    '''log image result'''
    if iteration_i % 500 == 0:

        '''log real result'''
        state_prediction_gt_one = state_prediction_gt[0]
        c = state_prediction_gt_one / 3.0
        c = torch.unsqueeze(c,1)
        state_prediction_gt_one_save = torch.cat([c,c,c],1)
        # state_prediction_gt_one_save = state_prediction_gt_one_save.mul(0.5).add(0.5)
        vutils.save_image(state_prediction_gt_one_save, '{0}/real_samples_{1}.png'.format(opt.experiment, iteration_i))

        '''log perdict result'''
        state_prediction_one = state_prediction[0]
        c = state_prediction_one / 3.0
        c = torch.unsqueeze(c,1)
        state_prediction_one_save = torch.cat([c,c,c],1)
        # state_prediction_one_save = state_prediction_one_save.mul(0.5).add(0.5)
        vutils.save_image(state_prediction_one_save, '{0}/fake_samples_{1}.png'.format(opt.experiment, iteration_i))

        '''do checkpointing'''
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, iteration_i))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, iteration_i))

    iteration_i += 1
    ######################################################################
    ######################### End One in Iteration  ######################
    ######################################################################

    
