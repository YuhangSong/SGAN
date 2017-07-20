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
parser.add_argument('--nc', type=int, default=config.gan_nc, help='input image channels')
parser.add_argument('--nz', type=int, default=config.gan_nz, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=config.gan_ngpu, help='number of GPUs to use')
parser.add_argument('--netG_Cv', default=config.modeldir+'netG_Cv.pth', help="path to netG_Cv (to continue training)")
parser.add_argument('--netG_DeCv', default=config.modeldir+'netG_DeCv.pth', help="path to netG_DeCv (to continue training)")
parser.add_argument('--netD', default=config.modeldir+'netD.pth', help="path to netD (to continue training)")
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

subprocess.call(["mkdir", "-p", config.logdir])
subprocess.call(["mkdir", "-p", config.modeldir])

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

'''create models'''
netG_Cv = dcgan.DCGAN_G_Cv(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netG_DeCv = dcgan.DCGAN_G_DeCv(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)

'''init models'''
netG_Cv.apply(weights_init)
netG_DeCv.apply(weights_init)
netD.apply(weights_init)

# do auto checkpoint
try:
    netG_Cv.load_state_dict(torch.load(opt.netG_Cv))
    print('Previous checkpoint for netG_Cv founded')
except Exception, e:
    print('Previous checkpoint for netG_Cv unfounded')
try:
    netG_DeCv.load_state_dict(torch.load(opt.netG_DeCv))
    print('Previous checkpoint for netG_DeCv founded')
except Exception, e:
    print('Previous checkpoint for netG_DeCv unfounded')
try:
    netD.load_state_dict(torch.load(opt.netD))
    print('Previous checkpoint for netD founded')
except Exception, e:
    print('Previous checkpoint for netD unfounded')

'''print the models'''
print(netG_Cv)
print(netG_DeCv)
print(netD)

inputd = torch.FloatTensor(opt.batchSize, 4, opt.imageSize, opt.imageSize)
inputd_real_part = torch.FloatTensor(opt.batchSize, 4, opt.imageSize, opt.imageSize)
inputg = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
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
    netG_Cv.cuda()
    netG_DeCv.cuda()
    inputd = inputd.cuda()
    inputg = inputg.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    # dataset = dataset.cuda()
    # dataset_sampler_indexs = dataset_sampler_indexs.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG_Cv = optim.RMSprop(netG_Cv.parameters(), lr = opt.lrG)
    optimizerG_DeCv = optim.RMSprop(netG_DeCv.parameters(), lr = opt.lrG)

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
        raw = torch.index_select(dataset,0,dataset_sampler_indexs.random_(0,dataset_len))
        image = [] 
        for image_i in range(4):
            image += [raw.narrow(1,image_i,1)]
        state_prediction_gt = torch.cat(image,2)
        state_prediction_gt = torch.squeeze(state_prediction_gt,1)
        if opt.cuda:
            state_prediction_gt = state_prediction_gt.cuda()
        state = state_prediction_gt.narrow(1,0*nc,3*nc)
        prediction_gt = state_prediction_gt.narrow(1,3*nc,1*nc)

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

        # compute encoded
        encodedv = netG_Cv(inputgv)

        # compute noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile = True) # totally freeze netG

        # concate encodedv and noisev
        encodedv_noisev = torch.cat([encodedv,noisev],1)

        # predict
        prediction = netG_DeCv(encodedv_noisev)
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
    netG_Cv.zero_grad()
    netG_DeCv.zero_grad()

    # feed
    inputg.resize_as_(state).copy_(state)
    inputgv = Variable(inputg)

    # compute encodedv
    encodedv = netG_Cv(inputgv)

    # compute noisev
    noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
    noisev = Variable(noise)

    # concate encodedv and noisev
    encodedv_noisev = torch.cat([encodedv,noisev],1)

    # predict
    prediction = netG_DeCv(encodedv_noisev)

    # get state_predictionv, this is a Variable cat 
    statev_predictionv = torch.cat([Variable(state), prediction], 1)

    # feed, this state_predictionv is Variable
    inputdv = statev_predictionv

    # compute
    errG, _ = netD(inputdv)
    errG.backward(one)

    # optmize
    optimizerG_Cv.step()
    optimizerG_DeCv.step()

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
    if iteration_i % 100 == 0:

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
        vutils.save_image(sample2image(state_prediction_gt[0]), '{0}/real_samples_{1}.png'.format(opt.experiment, iteration_i))

        '''log perdict result'''
        vutils.save_image(sample2image(state_prediction[0]), '{0}/fake_samples_{1}.png'.format(opt.experiment, iteration_i))

        '''do checkpointing'''
        torch.save(netG_Cv.state_dict(), '{0}/{1}/netG_Cv.pth'.format(opt.experiment,config.gan_model_name_))
        torch.save(netG_DeCv.state_dict(), '{0}/{1}/netG_DeCv.pth'.format(opt.experiment,config.gan_model_name_))
        torch.save(netD.state_dict(), '{0}/{1}/netD.pth'.format(opt.experiment,config.gan_model_name_))

    iteration_i += 1
    ######################################################################
    ######################### End One in Iteration  ######################
    ######################################################################

    
