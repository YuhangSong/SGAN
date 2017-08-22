import os, sys
sys.path.append(os.getcwd())
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.datasets
import tflib as lib
import tflib.plot
import cv2

import subprocess
from PIL import Image
import torchvision.utils as vutils
import visdom
vis = visdom.Visdom()
import time
import math
import domains.all_domains as chris_domain
import matplotlib.cm as cm

CLEAR_RUN = False
MULTI_RUN = 'h-0'
GPU = '0'

MULTI_RUN = MULTI_RUN + '|GPU:' + GPU
#-------reuse--device
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
if GPU!=None:
    import torch
    import torch.autograd as autograd
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    torch.manual_seed(4213)
    GPU = range(torch.cuda.device_count())
    print('Using GPU:'+str(GPU))

params = {}
params_seq = []
def add_parameters(**kwargs):
    global params_seq
    params_seq += kwargs.keys()
    params.update(kwargs)

'''domain settings'''
add_parameters(EXP = 'simple_gg')
add_parameters(DOMAIN = '1Dflip') # 1Dgrid, 1Dflip, 2Dgrid,
add_parameters(FIX_STATE = False)
add_parameters(REPRESENTATION = chris_domain.VECTOR) # chris_domain.SCALAR, chris_domain.VECTOR, chris_domain.IMAGE
add_parameters(GRID_SIZE = 20)

'''domain dynamic'''
if params['DOMAIN']=='1Dflip':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/params['GRID_SIZE']]*params['GRID_SIZE'])
    # add_parameters(GRID_ACTION_DISTRIBUTION = [0,0,1,0,0])

elif params['DOMAIN']=='1Dgrid':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/3.0,2.0/3.0])
    # add_parameters(GRID_ACTION_DISTRIBUTION = [1.0,0.0])

elif params['DOMAIN']=='2Dgrid':
    add_parameters(GRID_ACTION_DISTRIBUTION = [0.8, 0.0, 0.1, 0.1])
    add_parameters(OBSTACLE_POS_LIST = [])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    # add_parameters(OBSTACLE_POS_LIST = [])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.8, 0.0, 0.1, 0.1])
    # add_parameters(OBSTACLE_POS_LIST = [(2, 2)])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    # add_parameters(OBSTACLE_POS_LIST = [(2, 2)])

else:
    print(unsupport)

'''method settings'''
add_parameters(METHOD = 'grl') # tabular, bayes-net-learner, deterministic-deep-net, grl

add_parameters(GP_MODE = 'none-guide') # none-guide, use-guide, pure-guide
add_parameters(GP_GUIDE_FACTOR = 1.0)

add_parameters(INTERPOLATES_MODE = 'one') # auto, one
add_parameters(DELTA_T = 0.1)

add_parameters(SOFT_GP = False)
add_parameters(SOFT_GP_FACTOR = 3)

add_parameters(STABLE_MSE = None) # None

'''model settings'''
if params['REPRESENTATION']==chris_domain.SCALAR:
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = 128)
    add_parameters(LAMBDA = 0.1)
    add_parameters(TARGET_W_DISTANCE = 0.1)

elif params['REPRESENTATION']==chris_domain.VECTOR:
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = 128)
    add_parameters(LAMBDA = 5)
    add_parameters(TARGET_W_DISTANCE = 0.1)

elif params['REPRESENTATION']==chris_domain.IMAGE:
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = 128)
    add_parameters(LAMBDA = 10)
    add_parameters(TARGET_W_DISTANCE = 0.1)

else:
    print(unsupport)

add_parameters(BATCH_SIZE = 32)

if params['DOMAIN']=='marble':
    add_parameters(STATE_DEPTH = 3)

else:
    add_parameters(STATE_DEPTH = 1)

add_parameters(FEATURE = 3)

'''default domain settings generate'''
if params['DOMAIN']=='1Dflip':
    domain = chris_domain.BitFlip1D(
        length=params['GRID_SIZE'],
        mode=params['REPRESENTATION'],
        prob_dirs=params['GRID_ACTION_DISTRIBUTION'],
        fix_state=params['FIX_STATE']
    )

elif params['DOMAIN']=='1Dgrid':
    domain = chris_domain.Walk1D(
        length=params['GRID_SIZE'],
        prob_left=params['GRID_ACTION_DISTRIBUTION'][0],
        mode=params['REPRESENTATION'],
        fix_state=params['FIX_STATE']
    )

elif params['DOMAIN']=='2Dgrid':
    domain = chris_domain.Walk2D(
        width=params['GRID_SIZE'],
        height=params['GRID_SIZE'],
        prob_dirs=params['GRID_ACTION_DISTRIBUTION'],
        obstacle_pos_list=params['OBSTACLE_POS_LIST'],
        mode=params['REPRESENTATION'],
        should_wrap=True,
        fix_state=params['FIX_STATE']
    )

else:
    print(unsupport)

'''history settings'''
add_parameters(RUINER_MODE = 'none-r') # none-r, use-r, test-r
add_parameters(GAN_MODE = 'wgan-grad-panish') # wgan, wgan-grad-panish, wgan-gravity, wgan-decade
add_parameters(OPTIMIZER = 'Adam') # Adam, RMSprop
add_parameters(CRITIC_ITERS = 5)

add_parameters(AUX_INFO = 'no bn on vector and scalar domain')

'''summary settings'''
DSP = ''
params_str = 'Settings'+'\n'
params_str += '##################################'+'\n'
for i in range(len(params_seq)):
    DSP += params_seq[i]+'_'+str(params[params_seq[i]]).replace('.','_').replace(',','_').replace(' ','_')+'/'
    params_str += params_seq[i]+' >> '+str(params[params_seq[i]])+'\n'
params_str += '##################################'+'\n'
print(params_str)

BASIC = '../../result/'
LOGDIR = BASIC+DSP
if CLEAR_RUN:
    subprocess.call(["rm", "-r", LOGDIR])
subprocess.call(["mkdir", "-p", LOGDIR])
with open(LOGDIR+"Settings.txt","a") as f:
    f.write(params_str)

N_POINTS = 128
RESULT_SAMPLE_NUM = 1000
FILTER_RATE = 0.5
TrainTo   = 100000
LOG_INTER =   1000
if params['DOMAIN']=='1Dflip':
    if params['GRID_SIZE']>5:
        LOG_INTER = 10000
    if params['GRID_SIZE']>10:
        LOG_INTER = TrainTo

if params['REPRESENTATION']==chris_domain.SCALAR:
    if params['DOMAIN']=='2Dgrid':
        DESCRIBE_DIM = 2

    else:
        print(unsupport)

elif params['REPRESENTATION']==chris_domain.VECTOR:
    if params['DOMAIN']=='1Dgrid' or params['DOMAIN']=='1Dflip':
        DESCRIBE_DIM = params['GRID_SIZE']

    else:
        print(unsupport)

############################### Definition Start ###############################

def vector2image(x):
    block_size = chris_domain.BLOCK_SIZE*3
    x_temp = torch.FloatTensor(
        x.size()[0],
        x.size()[1],
        1,
        block_size,
        params['GRID_SIZE']*block_size
    ).cuda().fill_(0.0)
    for b in range(x.size()[0]):
        for d in range(x.size()[1]):
            for i in range(x.size()[2]):
                from_ = i*block_size
                to_ = (i+1)*block_size
                fill_ = float(x[b][d][i])
                x_temp[b,d,0,:,from_:to_].fill_(fill_)
    return x_temp

def log_img(x,name,iteration):
    if params['REPRESENTATION']==chris_domain.VECTOR:
        x = vector2image(x)
    x = x.squeeze(1)
    vutils.save_image(x, LOGDIR+name+'_'+str(iteration)+'.png')
    vis.images( x.cpu().numpy(),
                win=str(MULTI_RUN)+'-'+name,
                opts=dict(caption=str(MULTI_RUN)+'-'+name+'_'+str(iteration)))

def plt_to_vis(fig,win,name):
    canvas=fig.canvas
    import io
    buf = io.BytesIO()
    canvas.print_png(buf)
    data=buf.getvalue()
    buf.close()

    buf=io.BytesIO()
    buf.write(data)
    img=Image.open(buf)
    img = np.asarray(img)/255.0
    img = img.astype(float)[:,:,0:3]
    img = torch.FloatTensor(img).permute(2,0,1)
    vis.image(  img,
                win=str(MULTI_RUN)+'-'+win,
                opts=dict(title=str(MULTI_RUN)+'-'+name))

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            
            conv_layer = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.001),
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                nn.LeakyReLU(0.001),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001),
            )

            deconv_layer = nn.Sequential(
                nn.Linear(params['DIM'], DESCRIBE_DIM*(params['STATE_DEPTH']+1)),
                nn.Sigmoid()
            )

        elif params['REPRESENTATION']==chris_domain.IMAGE:

            conv_layer = nn.Sequential(
                # params['FEATURE']*1*25*25
                nn.Conv3d(
                    in_channels=params['FEATURE'],
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.001),
                # 64*1*12*12
                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.001),
                # 128*1*6*6
                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(0.001),
                # 256*1*3*3
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*3*3, params['DIM']),
                nn.LeakyReLU(0.001),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                nn.LeakyReLU(0.001),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], 256*1*3*3),
                nn.LeakyReLU(0.001),
            )
            deconv_layer = nn.Sequential(
                # 256*1*3*3
                nn.ConvTranspose3d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=(2,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.001),
                # 128*2*6*6
                nn.ConvTranspose3d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.001),
                # 64*2*12*12
                nn.ConvTranspose3d(
                    in_channels=64,
                    out_channels=params['FEATURE'],
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                    output_padding=(0,1,1)
                ),
                nn.Sigmoid()
                # params['FEATURE']*2*25*25
            )

        self.conv_layer = nn.DataParallel(conv_layer,GPU)
        self.squeeze_layer = nn.DataParallel(squeeze_layer,GPU)
        self.cat_layer = nn.DataParallel(cat_layer,GPU)
        self.unsqueeze_layer = nn.DataParallel(unsqueeze_layer,GPU)
        self.deconv_layer = torch.nn.DataParallel(deconv_layer,GPU)
        

    def forward(self, noise_v, state_v):

        '''prepare'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            state_v = state_v.squeeze(1)
        elif params['REPRESENTATION']==chris_domain.IMAGE:
            # N*D*F*H*W to N*F*D*H*W
            state_v = state_v.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(state_v)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            temp = x.size()
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.cat_layer(torch.cat([x,noise_v],1))
        x = self.unsqueeze_layer(x)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            x = x.view(temp)
        x = self.deconv_layer(x)

        '''decompose'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            stater_v = x.narrow(1,0,DESCRIBE_DIM*params['STATE_DEPTH']).unsqueeze(1)
            prediction_v = x.narrow(1,DESCRIBE_DIM*params['STATE_DEPTH'],DESCRIBE_DIM).unsqueeze(1)
            x = torch.cat([stater_v,prediction_v],1)

        elif params['REPRESENTATION']==chris_domain.IMAGE:
            # N*F*D*H*W to N*D*F*H*W
            x = x.permute(0,2,1,3,4)

        return x

class Transitor(nn.Module):

    def __init__(self):
        super(Transitor, self).__init__()

        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            
            conv_layer = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(params['DIM']),
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(params['DIM']),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(params['DIM']),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(params['DIM']),
            )
            if params['REPRESENTATION']==chris_domain.SCALAR:
                deconv_layer = nn.Sequential(
                    nn.Linear(params['DIM'], DESCRIBE_DIM*(params['STATE_DEPTH']+1))
                )
            elif params['REPRESENTATION']==chris_domain.VECTOR:
                deconv_layer = nn.Sequential(
                    nn.Linear(params['DIM'], DESCRIBE_DIM*(params['STATE_DEPTH']+1)),
                    nn.Sigmoid()
                )

        elif params['REPRESENTATION']==chris_domain.IMAGE:

            conv_layer = nn.Sequential(
                # params['FEATURE']*1*32*32
                nn.Conv3d(
                    in_channels=params['FEATURE'],
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # 64*1*16*16
                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # 128*1*8*8
                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # 256*1*4*4
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*4*4, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], 256*1*4*4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            deconv_layer = nn.Sequential(
                # 256*1*4*4
                nn.ConvTranspose3d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=(2,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # 128*2*8*8
                nn.ConvTranspose3d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # 64*2*16*16
                nn.ConvTranspose3d(
                    in_channels=64,
                    out_channels=params['FEATURE'],
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.Sigmoid()
                # params['FEATURE']*2*32*32  
            )

        self.conv_layer = nn.DataParallel(conv_layer,GPU)
        self.squeeze_layer = nn.DataParallel(squeeze_layer,GPU)
        self.cat_layer = nn.DataParallel(cat_layer,GPU)
        self.unsqueeze_layer = nn.DataParallel(unsqueeze_layer,GPU)
        self.deconv_layer = torch.nn.DataParallel(deconv_layer,GPU)
        

    def forward(self, state_v):

        '''prepare'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            state_v = state_v.squeeze(1)
        elif params['REPRESENTATION']==chris_domain.IMAGE:
            # N*D*F*H*W to N*F*D*H*W
            state_v = state_v.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(state_v)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            temp = x.size()
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.cat_layer(x)
        x = self.unsqueeze_layer(x)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            x = x.view(temp)
        x = self.deconv_layer(x)

        '''decompose'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            stater_v = x.narrow(1,0,DESCRIBE_DIM*params['STATE_DEPTH']).unsqueeze(1)
            prediction_v = x.narrow(1,DESCRIBE_DIM*params['STATE_DEPTH'],DESCRIBE_DIM).unsqueeze(1)
            x = torch.cat([stater_v,prediction_v],1)

        elif params['REPRESENTATION']==chris_domain.IMAGE:
            # N*F*D*H*W to N*D*F*H*W
            x = x.permute(0,2,1,3,4)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:

            conv_layer_state = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            squeeze_layer_state = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            conv_layer_prediction = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            squeeze_layer_prediction = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            final_layer = nn.Sequential(
                nn.Linear(params['DIM']*2, params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
                nn.Linear(params['DIM'], 1),
            )

        elif params['REPRESENTATION']==chris_domain.IMAGE:

            conv_layer = nn.Sequential(
                # params['FEATURE']*2*25*25
                nn.Conv3d(
                    in_channels=params['FEATURE'],
                    out_channels=64,
                    kernel_size=(2,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.001, inplace=True),
                # 64*1*12*12
                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.001, inplace=True),
                # 128*1*6*6
                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.001, inplace=True),
                # 256*1*3*3
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*3*3, params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            final_layer = nn.Sequential(
                nn.Linear(params['DIM'], 1),
            )

        else:
            raise Exception('Unsupport')

        if params['GAN_MODE']=='wgan-grad-panish':

            if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
                self.conv_layer_state = conv_layer_state
                self.squeeze_layer_state = squeeze_layer_state
                self.conv_layer_prediction = conv_layer_prediction
                self.squeeze_layer_prediction = squeeze_layer_prediction

            elif params['REPRESENTATION']==chris_domain.IMAGE:
                self.conv_layer = conv_layer
                self.squeeze_layer = squeeze_layer

            else:
                raise Exception('Unsupport')

            self.final_layer = final_layer

        else:

            if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
                self.conv_layer_state = torch.nn.DataParallel(conv_layer_state,GPU)
                self.squeeze_layer_state = torch.nn.DataParallel(squeeze_layer_state,GPU)
                self.conv_layer_prediction = torch.nn.DataParallel(conv_layer_prediction,GPU)
                self.squeeze_layer_prediction = torch.nn.DataParallel(squeeze_layer_prediction,GPU)

            elif params['REPRESENTATION']==chris_domain.IMAGE:
                self.conv_layer = torch.nn.DataParallel(conv_layer,GPU)
                self.squeeze_layer = torch.nn.DataParallel(squeeze_layer,GPU)

            else:
                raise Exception('Unsupport')

            self.final_layer = torch.nn.DataParallel(final_layer,GPU)

    def forward(self, state_v, prediction_v):

        '''prepare'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:

            state_v = state_v.squeeze(1)
            prediction_v = prediction_v.squeeze(1)

            state_v = self.conv_layer_state(state_v)
            state_v = self.squeeze_layer_state(state_v)

            prediction_v = self.conv_layer_prediction(prediction_v)
            prediction_v = self.squeeze_layer_prediction(prediction_v)

            x = torch.cat([state_v,prediction_v],1)
            x = self.final_layer(x)
            x = x.view(-1)

        elif params['REPRESENTATION']==chris_domain.IMAGE:
            x = torch.cat([state_v,prediction_v],1)
            # N*D*F*H*W to N*F*D*H*W
            x = x.permute(0,2,1,3,4)

            '''forward'''
            x = self.conv_layer(x)
            if params['REPRESENTATION']==chris_domain.IMAGE:
                x = x.view(x.size()[0], -1)
            x = self.squeeze_layer(x)
            x = self.final_layer(x)
            x = x.view(-1)

        return x

class Corrector(nn.Module):

    def __init__(self):
        super(Corrector, self).__init__()

        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:

            conv_layer = nn.Sequential(
                nn.Linear(2*DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            final_layer = nn.Sequential(
                nn.Linear(params['DIM'], 1),
                nn.Sigmoid(),
            )

        elif params['REPRESENTATION']==chris_domain.IMAGE:

            conv_layer =    nn.Sequential(
                # params['FEATURE']*2*32*32
                nn.Conv3d(
                    in_channels=params['FEATURE'],
                    out_channels=64,
                    kernel_size=(2,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True),
                # 64*1*16*16
                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True),
                # 128*1*8*8
                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True),
                # 256*1*4*4
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*4*4, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            final_layer = nn.Sequential(
                nn.Linear(params['DIM'], 1),
                nn.Sigmoid(),
            )

        self.conv_layer = torch.nn.DataParallel(conv_layer,GPU)
        self.squeeze_layer = torch.nn.DataParallel(squeeze_layer,GPU)
        self.final_layer = torch.nn.DataParallel(final_layer,GPU)

    def forward(self, state_v, prediction_v):

        '''prepare'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            state_v = state_v.squeeze(1)
            prediction_v = prediction_v.squeeze(1)
            x = torch.cat([state_v,prediction_v],1)
        elif params['REPRESENTATION']==chris_domain.IMAGE:
            x = torch.cat([state_v,prediction_v],1)
            # N*D*F*H*W to N*F*D*H*W
            x = x.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(x)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.final_layer(x)
        x = x.view(-1)

        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(
            m.weight.data,
            gain=1
        )
        m.bias.data.fill_(0.1)
    # elif classname.find('Conv3d') != -1:
    #     torch.nn.init.xavier_uniform(
    #         m.weight.data,
    #         gain=1
    #     )
    # elif classname.find('ConvTranspose3d') != -1:
    #     torch.nn.init.xavier_uniform(
    #         m.weight.data,
    #         gain=1
    #     )

def chris2song(x):

    if (params['REPRESENTATION']==chris_domain.SCALAR) or (params['REPRESENTATION']==chris_domain.VECTOR):
        x = torch.from_numpy(np.array(x)).cuda().unsqueeze(1).float()

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        x = torch.from_numpy(np.array(x)).cuda().unsqueeze(1).permute(0,1,4,2,3).float()/255.0

    else:
        raise Exception('ss')
    
    return x

def song2chris(x):

    if (params['REPRESENTATION']==chris_domain.SCALAR) or (params['REPRESENTATION']==chris_domain.VECTOR):
        x = x.squeeze(1).cpu().numpy()

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        x = (x*255.0).byte().permute(0,1,3,4,2).squeeze(1).cpu().numpy()

    else:
        raise Exception('ss')

    return x

def get_tabular_samples(tabular_dic,start_state):

    for tabular_i in tabular_dic:
        delta = np.mean(
            np.abs(start_state - tabular_i.x),
            keepdims=False
        )
        if delta==0.0:
            print('Found tabular.')
            total = 0
            for x_next_i in tabular_i.x_next_dic:
                total += x_next_i.count
            samples = []
            for x_next_i in tabular_i.x_next_dic:
                if x_next_i.count!=0:
                    samples += [x_next_i.x]*long((float(x_next_i.count)/total*RESULT_SAMPLE_NUM))
            return np.array(samples).astype(float)

def collect_samples(iteration,tabular=None):

    all_possible = chris2song(domain.get_all_possible_start_states())

    all_l1 = []
    all_ac = []
    for ii in range(all_possible.size()[0]):

        start_state = all_possible[ii:ii+1]
        
        if tabular is None:

            if (params['REPRESENTATION']==chris_domain.SCALAR) or (params['REPRESENTATION']==chris_domain.VECTOR):
                start_state_batch = start_state.repeat(RESULT_SAMPLE_NUM,1,1)

            elif params['REPRESENTATION']==chris_domain.IMAGE:
                start_state_batch = start_state.repeat(RESULT_SAMPLE_NUM,1,1,1,1)

            else:
                raise Exception('ss')
            
            if params['METHOD']=='deterministic-deep-net':
                prediction = netT(
                    state_v = autograd.Variable(start_state_batch)
                ).narrow(1,(params['STATE_DEPTH']-1),1).data

            elif params['METHOD']=='grl':
                '''prediction'''
                noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
                prediction = netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(start_state_batch, volatile=True)
                ).data.narrow(1,params['STATE_DEPTH'],1)

            else:
                raise Exception('Unsupport')

        else:
            
            prediction = get_tabular_samples(tabular,start_state[0].cpu().numpy())
            
            if prediction is not None:
                prediction = torch.cuda.FloatTensor(prediction)

        if ii==all_possible.size()[0]/2 and prediction is not None:
            log_img(prediction,'prediction',iteration)

        if prediction is not None:
            l1, ac = chris_domain.evaluate_domain(
                domain=domain,
                s1_state=song2chris(start_state)[0],
                s2_samples=song2chris(prediction)
            )

        else:
            l1, ac = 2.0, 0.0

        all_l1.append(l1)
        all_ac.append(ac)

        print('[{}][{}][Eval:{}/{}] L1:{:2.4f} AC:{:2.4f}'
            .format(
                MULTI_RUN,
                iteration,
                ii,
                all_possible.size()[0],
                l1,
                ac
            )
        )

    return np.mean(all_l1), np.mean(all_ac)

def generate_image(iteration,tabular=None):

    if params['REPRESENTATION']==chris_domain.SCALAR:

        batch_size = (N_POINTS**2)

        domain.set_fix_state(True)
        data_fix_state = dataset_iter(
            fix_state=True,
            batch_size=batch_size
        )
        dataset = data_fix_state.next()
        domain.set_fix_state(params['FIX_STATE'])

        generate_image_with_filter(
            iteration=iteration,
            dataset=dataset,
            gen_basic=True,
            filter_net=None
        )

    l1, accept_rate = collect_samples(iteration,tabular)

    logger.plot(
        '-L1',
        np.asarray([l1])
    )
    logger.plot(
        '-AR',
        np.asarray([accept_rate])
    )

    return l1, accept_rate

def plot_convergence(images,name):
    dis, accept_rate = get_transition_prob_distribution(images)
    if not (np.sum(dis)==0.0):
        kl = scipy.stats.entropy(
            dis,
            qk=params['GRID_ACTION_DISTRIBUTION'],
            base=None
        )
        logger.plot(
            name+'-KL',
            np.asarray([kl])
        )
    l1 = np.squeeze(np.sum(np.abs(dis - np.asarray(params['GRID_ACTION_DISTRIBUTION']))))
    logger.plot(
        name+'-L1',
        np.asarray([l1])
    )
    logger.plot(
        name+'-AR',
        np.asarray([accept_rate])
    )

def generate_image_with_filter(iteration,dataset,gen_basic=False,filter_net=None):

    plt.clf()

    state_prediction_gt = dataset
    state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
    prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

    '''disc_map'''
    if params['REPRESENTATION']==chris_domain.SCALAR:

        plt.title(MULTI_RUN+'@'+str(iteration))

        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(0, 1.0, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(0, 1.0, N_POINTS)[None, :]
        points = points.reshape((-1, 2))
        points = np.expand_dims(points,1)

        disc_map =  netD(
                        state_v = autograd.Variable(state, volatile=True),
                        prediction_v = autograd.Variable(torch.Tensor(points).cuda(), volatile=True)
                    ).cpu().data.numpy()
        x = y = np.linspace(0, 1.0, N_POINTS)
        disc_map = disc_map.reshape((len(x), len(y))).transpose()
        im = plt.imshow(disc_map, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(0, 1.0, 0, 1.0))
        levels = np.arange(-1.0, 1.0, 0.1)
        
        CS = plt.contour(x, y, disc_map, levels)

        plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=14)

        '''narrow to normal batch size'''
        state_prediction_gt = state_prediction_gt.narrow(0,0,RESULT_SAMPLE_NUM)
        state = state.narrow(0,0,RESULT_SAMPLE_NUM)
        prediction_gt = prediction_gt.narrow(0,0,RESULT_SAMPLE_NUM)
        
    prediction_gt_mean = prediction_gt.mean(0,keepdim=True)

    '''prediction_gt'''
    if params['REPRESENTATION']==chris_domain.SCALAR:
        plt.scatter(
            prediction_gt.squeeze(1).cpu().numpy()[:, 0], 
            prediction_gt.squeeze(1).cpu().numpy()[:, 1],
            c='orange', 
            marker='+', 
            alpha=0.5
        )
    else:
        if gen_basic:
            log_img(prediction_gt,'prediction_gt',iteration)
            prediction_gt_mean = prediction_gt.mean(0,keepdim=True)
            log_img(prediction_gt_mean,'prediction_gt_mean',iteration)
            plot_convergence(
                images=prediction_gt,
                name='prediction_gt_mean'
            )

    '''prediction_gt_r'''
    if params['RUINER_MODE']!='none-r':
        if params['REPRESENTATION']==chris_domain.SCALAR:
            noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
            prediction_gt_r = netG(
                noise_v = autograd.Variable(noise, volatile=True),
                state_v = autograd.Variable(prediction_gt, volatile=True)
            ).data.narrow(1,0,1)
            plt.scatter(
                prediction_gt_r.squeeze(1).cpu().numpy()[:, 0], 
                prediction_gt_r.squeeze(1).cpu().numpy()[:, 1], 
                c='blue', 
                marker='+', 
                alpha=0.5
            )
        else:
            if gen_basic:
                noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
                prediction_gt_r = netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(prediction_gt, volatile=True)
                ).data.narrow(1,0,1)
                log_img(prediction_gt_r,'prediction_gt_r',iteration)
                prediction_gt_r_mean = prediction_gt_r.mean(0,keepdim=True)
                log_img(prediction_gt_r_mean,'prediction_gt_r_mean',iteration)
                plot_convergence(
                    images=prediction_gt_r,
                    name='prediction_gt_r_mean'
                )

    '''prediction'''
    noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
    prediction = netG(
        noise_v = autograd.Variable(noise, volatile=True),
        state_v = autograd.Variable(state, volatile=True)
    ).data.narrow(1,params['STATE_DEPTH'],1)
    
    if filter_net is not None:
        F_out = filter_net(
            state_v = autograd.Variable(state, volatile=True),
            prediction_v = autograd.Variable(prediction, volatile=True)
        ).data
        normal_f = (torch.max(F_out)-torch.min(F_out))
        if normal_f==0.0:
            print('Invalid test, return.')
            return
        F_out = (F_out - torch.min(F_out)) / normal_f

    if params['REPRESENTATION']==chris_domain.SCALAR:
        plt.scatter(
            prediction.squeeze(1).cpu().numpy()[:, 0], 
            prediction.squeeze(1).cpu().numpy()[:, 1], 
            c='green', 
            marker='+', 
            alpha=0.5
        )
        if filter_net is not None:
            F_out = F_out.cpu().numpy()
            for ii in range(prediction.size()[0]):
                plt.scatter(
                    prediction.squeeze(1).cpu().numpy()[ii, 0],
                    prediction.squeeze(1).cpu().numpy()[ii, 1], 
                    c='red', 
                    marker='s', 
                    alpha=F_out[ii]
                )
        gradient_penalty, num_t_sum = calc_gradient_penalty(
            netD = netD,
            state = state,
            prediction = prediction,
            prediction_gt = prediction_gt,
            log=True
        )
    else:

        if filter_net is None:

            log_img(prediction,'prediction',iteration)
            prediction_mean = prediction.mean(0,keepdim=True)
            log_img(
                prediction_mean,
                'prediction-mean',
                iteration
            )
            plot_convergence(
                images=prediction,
                name='prediction-non-filtered'
            )

        else:

            F_out_numpy = F_out.cpu().numpy()

            while len(F_out.size())!=len(prediction.size()):
                F_out = F_out.unsqueeze(1)
            if params['REPRESENTATION']==chris_domain.VECTOR:
                F_out = F_out.repeat(
                    1,
                    prediction.size()[1],
                    prediction.size()[2])
            elif params['REPRESENTATION']==chris_domain.IMAGE:
                F_out = F_out.repeat(
                    1,
                    prediction.size()[1],
                    prediction.size()[2],
                    prediction.size()[3],
                    prediction.size()[4]
                )

            log_img(
                x=F_out,
                name='prediction-filter-'+str(filter_net.__class__.__name__),
                iteration=iteration
            )

            indexs = np.array(range(np.shape(F_out_numpy)[0]))
            F_out_numpy_indexs = np.concatenate(
                (np.expand_dims(
                    F_out_numpy,
                    axis=1),
                np.expand_dims(
                    indexs,
                    axis=1)),
                axis=1
            )
            F_out_numpy_indexs = F_out_numpy_indexs[F_out_numpy_indexs[:,0].argsort(kind='mergesort')]
            filter_num = int(RESULT_SAMPLE_NUM*FILTER_RATE)
            filtered_indexs = F_out_numpy_indexs[RESULT_SAMPLE_NUM-filter_num:RESULT_SAMPLE_NUM,1]
            filtered_indexs = filtered_indexs.astype(int)
            filtered_prediction = torch.index_select(prediction,0,torch.cuda.LongTensor(filtered_indexs))
            log_img(
                x=filtered_prediction,
                name='prediction-filtered-by-'+str(filter_net.__class__.__name__),
                iteration=iteration
            )
            filtered_prediction_mean = filtered_prediction.mean(0,keepdim=True)
            log_img(
                x=filtered_prediction_mean,
                name='prediction-mean-filtered-by-'+str(filter_net.__class__.__name__),
                iteration=iteration
            )

            plot_convergence(
                images=filtered_prediction,
                name='prediction-filtered-by-'+str(filter_net.__class__.__name__)
            )

    if params['REPRESENTATION']==chris_domain.SCALAR:
        if filter_net is None:
            file_name = ''
        else:
            file_name = 'filtered-by-'+str(filter_net.__class__.__name__)
        plt.savefig(LOGDIR+file_name+'_'+str(iteration)+'.jpg')
        plt_to_vis(
            plt.gcf(),
            win=file_name,
            name=file_name+'_'+str(iteration))

def get_state(fix_state=False):

    if not fix_state:
        cur_x = np.random.choice(range(params['GRID_SIZE']))
        cur_y = np.random.choice(range(params['GRID_SIZE']))
    else:
        cur_x = FIX_STATE_TO[0]
        cur_y = FIX_STATE_TO[1]

    return cur_x,cur_y

def transition(cur_x,cur_y,action):

    next_x = cur_x
    next_y = cur_y
    if action==0:
        next_x = cur_x + 1
    elif action==1:
        next_y = cur_y + 1
    elif action==2:
        next_x = cur_x - 1
    elif action==3:
        next_y = cur_y - 1
    next_x = np.clip(next_x,0,params['GRID_SIZE']-1)
    next_y = np.clip(next_y,0,params['GRID_SIZE']-1)
    return next_x,next_y

def get_ob(x,y):

    ob = [x,y]
    ob = np.asarray(ob).astype(float) / (params['GRID_SIZE']-1)
    ob = np.expand_dims(ob,0)

    return ob

def dataset_iter(fix_state=False, batch_size=params['BATCH_SIZE']):

    while True:

        dataset = None

        for i in xrange(batch_size):

            domain.reset()
            ob = torch.from_numpy(domain.get_state()).cuda().unsqueeze(0)
            ob_next = torch.from_numpy(domain.update()).cuda().unsqueeze(0)

            data = torch.cat([ob,ob_next],0).unsqueeze(0)

            try:
                dataset = torch.cat([dataset,data],0)
            except Exception as e:
                dataset = data

        if params['REPRESENTATION']==chris_domain.SCALAR:
            dataset = dataset.float()
        if params['REPRESENTATION']==chris_domain.VECTOR:
            dataset = dataset.float()
        elif params['REPRESENTATION']==chris_domain.IMAGE:
            dataset = dataset.permute(0,1,4,2,3)
            dataset = dataset.float()
            dataset = dataset / 255.0

        yield dataset

def calc_gradient_penalty(netD, state, prediction, prediction_gt, log=False):
    
    '''get multiple interplots'''
    if params['INTERPOLATES_MODE']=='auto':
        prediction_fl = prediction.contiguous().view(prediction.size()[0],-1)
        prediction_gt_fl = prediction_gt.contiguous().view(prediction_gt.size()[0],-1)
        max_norm = (prediction_gt_fl.size()[1])**0.5      
        d_mean = (prediction_gt_fl-prediction_fl).norm(2,dim=1)/max_norm
        num_t = (d_mean / params['DELTA_T']).floor().int() - 1

        num_t_sum = 0.0
        for b in range(num_t.size()[0]):

            if num_t[b]<1:
                continue
            else:
                t = num_t[b]
                num_t_sum += t

            if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
                state_b = state[b].unsqueeze(0).repeat(t,1,1)
                prediction_b = prediction[b].unsqueeze(0).repeat(t,1,1)
                prediction_gt_b = prediction_gt[b].unsqueeze(0).repeat(t,1,1)
            elif params['REPRESENTATION']==chris_domain.IMAGE:
                state_b = state[b].unsqueeze(0).repeat(t,1,1,1,1)
                prediction_b = prediction[b].unsqueeze(0).repeat(t,1,1,1,1)
                prediction_gt_b = prediction_gt[b].unsqueeze(0).repeat(t,1,1,1,1)
            
            try:
                state_delted = torch.cat(
                    [state_delted,state_b],
                    0
                )
                prediction_delted = torch.cat(
                    [prediction_delted,prediction_b],
                    0
                )
                prediction_gt_delted = torch.cat(
                    [prediction_gt_delted,prediction_gt_b],
                    0
                )
            except Exception as e:
                state_delted = state_b
                prediction_delted = prediction_b
                prediction_gt_delted = prediction_gt_b

        if num_t_sum > 0:
            state = state_delted
            prediction = prediction_delted
            prediction_gt = prediction_gt_delted
            alpha = torch.rand(prediction_gt.size()[0]).cuda()

        else:
            return None, num_t_sum

    else:
        num_t_sum = 1.0
        alpha = torch.rand(prediction_gt.size()[0]).cuda()

    if params['SOFT_GP']:
        '''solf function here'''
        alpha = (alpha*params['SOFT_GP_FACTOR']).tanh()

    while len(alpha.size())!=len(prediction_gt.size()):
        alpha = alpha.unsqueeze(1)

    if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
        alpha = alpha.repeat(
            1,
            prediction_gt.size()[1],
            prediction_gt.size()[2]
        )

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        alpha = alpha.repeat(
            1,
            prediction_gt.size()[1],
            prediction_gt.size()[2],
            prediction_gt.size()[3],
            prediction_gt.size()[4]
        )

    else:
        raise Exception('Unsupport')

    interpolates = ((1.0 - alpha) * prediction_gt) + (alpha * prediction)

    if log:
        plt.scatter(
            interpolates.squeeze(1).cpu().numpy()[:, 0], 
            interpolates.squeeze(1).cpu().numpy()[:, 1],
            c='red', 
            marker='+', 
            alpha=0.1
        )

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(
                            state_v = autograd.Variable(state),
                            prediction_v = interpolates
                        )

    gradients = autograd.grad(
                    outputs=disc_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
    gradients = gradients.contiguous()
    gradients_fl = gradients.view(gradients.size()[0],-1)

    if (params['GP_MODE']=='use-guide') or (params['GP_MODE']=='pure-guide'):

        prediction_fl = prediction.contiguous().view(prediction.size()[0],-1)
        prediction_gt_fl = prediction_gt.contiguous().view(prediction_gt.size()[0],-1)

        gradients_direction_gt_fl = prediction_gt_fl - prediction_fl
        gradients_direction_gt_fl = gradients_direction_gt_fl/(gradients_direction_gt_fl.norm(2,dim=1).unsqueeze(1).repeat(1,gradients_direction_gt_fl.size()[1]))

        gradients_direction_gt_fl = autograd.Variable(gradients_direction_gt_fl)

        gradients_penalty = (gradients_fl-gradients_direction_gt_fl).norm(2,dim=1).pow(2).mean()

        if params['GP_MODE']=='use-guide':
            gradients_penalty = gradients_penalty * params['LAMBDA'] * params['GP_GUIDE_FACTOR']

        if math.isnan(gradients_penalty.data.cpu().numpy()[0]):
            print('Bad gradients_penalty, return!')
            gradients_penalty = None

    elif params['GP_MODE']=='none-guide':

        gradients_penalty = ((gradients_fl.norm(2, dim=1) - 1.0) ** 2).mean()

        gradients_penalty = gradients_penalty * params['LAMBDA']
    
    else:
        raise Exception('Unsupport')
    
    return gradients_penalty, num_t_sum

def restore_model():
    print('Trying load models....')
    try:
        netD.load_state_dict(torch.load('{0}/netD.pth'.format(LOGDIR)))
        print('Previous checkpoint for netD founded')
    except Exception, e:
        print('Previous checkpoint for netD unfounded')
    try:
        netC.load_state_dict(torch.load('{0}/netC.pth'.format(LOGDIR)))
        print('Previous checkpoint for netC founded')
    except Exception, e:
        print('Previous checkpoint for netC unfounded')
    try:
        netG.load_state_dict(torch.load('{0}/netG.pth'.format(LOGDIR)))
        print('Previous checkpoint for netG founded')
    except Exception, e:
        print('Previous checkpoint for netG unfounded')
    print('')

class TabulatCell(object):
    """docstring for TabulatCell"""
    def __init__(self, x):
        super(TabulatCell, self).__init__()
        self.x = x
        self.count = 0.0
    def push(self):
        self.count += 1.0

class Tabular(object):
    """docstring for Tabular"""
    def __init__(self, x):
        super(Tabular, self).__init__()
        self.x = x
        self.x_next_dic = []
    def push(self,x_next_push):
        in_cell = False
        for x_next in self.x_next_dic:
            delta = np.mean(
                np.abs((x_next_push-x_next.x)),
                keepdims=False
            )
            if delta==0.0:
                x_next.push()
                in_cell = True
                break
        
        if not in_cell:
            self.x_next_dic += [TabulatCell(np.copy(x_next_push))]
            # print('Create a cell.')
            self.x_next_dic[-1].push()
            print(len(self.x_next_dic))
        
############################### Definition End ###############################

if params['METHOD']=='tabular':
    tabular_dic = []
    l1 = 2.0

elif params['METHOD']=='deterministic-deep-net':
    netT = Transitor().cuda()
    netT.apply(weights_init)
    print netT

    if params['OPTIMIZER']=='Adam':
        optimizerT = optim.Adam(netT.parameters(), lr=1e-4, betas=(0.5, 0.9))
    elif params['OPTIMIZER']=='RMSprop':
        optimizerT = optim.RMSprop(netT.parameters(), lr = 0.00005)

    mse_loss_model = torch.nn.MSELoss(size_average=True)

    data_fix_state = dataset_iter(
        fix_state=True,
        batch_size=1
    )

    L1, AC = 2.0, 0.0

elif params['METHOD']=='grl':

    netG = Generator().cuda()
    netD = Discriminator().cuda()
    netC = Corrector().cuda()

    netD.apply(weights_init)
    netC.apply(weights_init)
    netG.apply(weights_init)
    print netG
    print netD
    print netC

    if params['OPTIMIZER']=='Adam':
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optimizerC = optim.Adam(netC.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.0, 0.9))
    elif params['OPTIMIZER']=='RMSprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr = 0.00005)
        optimizerC = optim.RMSprop(netC.parameters(), lr = 0.00005)
        optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)

    mse_loss_model = torch.nn.MSELoss(size_average=True)

    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    ones_zeros = torch.cuda.FloatTensor(np.concatenate((np.ones((params['BATCH_SIZE'])),np.zeros((params['BATCH_SIZE']))),0))

    restore_model()

    L1, AC = 2.0, 0.0

    if params['STABLE_MSE'] is not None:
        stabling_mse = True
    else:
        stabling_mse = False

data = dataset_iter()

logger = lib.plot.logger(LOGDIR,DSP,params_str,MULTI_RUN)
iteration = logger.restore()

while True:
    iteration += 1

    if params['METHOD']=='tabular':

        '''get data set'''
        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH']).cpu().numpy()
        prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1).cpu().numpy()

        for b in range(np.shape(state)[0]):
            in_tabular = False
            for tabular_i in tabular_dic:
                delta = np.mean(
                    np.abs(state[b] - tabular_i.x),
                    keepdims=False
                )
                if delta==0.0:
                    tabular_i.push(np.copy(prediction_gt[b]))
                    in_tabular = True
                    break
            if not in_tabular:
                tabular_dic += [Tabular(np.copy(state[b]))]
                tabular_dic[-1].push(np.copy(prediction_gt[b]))
                print('Create a tabular: '+str(len(tabular_dic)))

        if iteration % LOG_INTER == 5:
            l1,_ = generate_image(iteration,tabular_dic)

        print('[{}][{}] l1: {}'
            .format(
                MULTI_RUN,
                iteration,
                l1
            )
        )

    elif params['METHOD']=='deterministic-deep-net':

        '''get data set'''
        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
        prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

        netT.zero_grad()
        
        prediction = netT(
            state_v = autograd.Variable(state)
        ).narrow(1,(params['STATE_DEPTH']-1),1)

        T = mse_loss_model(prediction, autograd.Variable(prediction_gt))

        T.backward()

        T_cost = T.data.cpu().numpy()
        logger.plot('T_cost', T_cost)
        
        optimizerT.step()

        if iteration % LOG_INTER == 5:
            L1, AC = generate_image(iteration)

        print('[{}][{:<6}] T_cost:{:2.4f} L1::{:2.4f} AC::{:2.4f}'
            .format(
                MULTI_RUN,
                iteration,
                T_cost[0],
                L1,
                AC
            )
        )

    elif params['METHOD']=='grl':

        if not stabling_mse:

            ########################################################
            ############### (1) Update D network ###################
            ########################################################

            for p in netD.parameters():
                p.requires_grad = True

            d_iter_num = params['CRITIC_ITERS']

            for iter_d in xrange(d_iter_num):

                if params['GAN_MODE']=='wgan':
                    '''
                    original wgan clip the weights of netD
                    '''
                    for p in netD.parameters():
                        p.data.clamp_(-0.01, +0.01)

                '''get data set'''
                state_prediction_gt = data.next()
                state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
                prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

                '''get generated data'''
                noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                prediction = netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(state, volatile=True)
                ).data.narrow(1,params['STATE_DEPTH'],1)

                if params['RUINER_MODE']=='use-r':
                    '''
                    if use-r, prediction is processed be r
                    '''
                    noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                    prediction_gt = netG(
                        noise_v = autograd.Variable(noise, volatile=True),
                        state_v = autograd.Variable(prediction_gt, volatile=True)
                    ).data.narrow(1,params['STATE_DEPTH']-1,1)
                    state_prediction_gt = torch.cat([state,prediction_gt],1)

                '''
                call backward in the following,
                prepare it.
                '''
                netD.zero_grad()

                '''if pure-guide, no wgan poss is used'''
                if not (params['GP_MODE']=='pure-guide'):
                    '''train with real'''
                    D_real = netD(
                        state_v = autograd.Variable(state),
                        prediction_v = autograd.Variable(prediction_gt)
                    ).mean()
                    D_real.backward(mone)

                else:
                    D_real = autograd.Variable(torch.cuda.FloatTensor([0.0]))

                '''if pure-guide, no wgan poss is used'''
                if not (params['GP_MODE']=='pure-guide'): 
                    '''train with fake'''
                    D_fake = netD(
                        state_v = autograd.Variable(state),
                        prediction_v = autograd.Variable(prediction)
                    ).mean()             
                    D_fake.backward(one)

                else:
                    D_fake = autograd.Variable(torch.cuda.FloatTensor([0.0]))

                GP_cost = [0.0]
                if params['GAN_MODE']=='wgan-grad-panish':
                    gradient_penalty, num_t_sum = calc_gradient_penalty(
                        netD = netD,
                        state = state,
                        prediction = prediction,
                        prediction_gt = prediction_gt
                    )

                    if gradient_penalty is not None:
                        gradient_penalty.backward()

                    else:
                        gradient_penalty = autograd.Variable(torch.cuda.FloatTensor([0.0]))

                    GP_cost = gradient_penalty.data.cpu().numpy()

                if params['GAN_MODE']=='wgan-grad-panish':
                    D_cost = D_fake - D_real + gradient_penalty
                elif params['GAN_MODE']=='wgan-decade':
                    D_cost = D_fake - D_real + decade_cost
                else:
                    D_cost = D_fake - D_real
                D_cost = D_cost.data.cpu().numpy()

                Wasserstein_D = (D_real - D_fake).data.cpu().numpy()

                optimizerD.step()

            if params['GAN_MODE']=='wgan-grad-panish':
                logger.plot('GP_cost', GP_cost)
            logger.plot('D_cost', D_cost)
            logger.plot('W_dis', Wasserstein_D)
            logger.plot('num_t_sum', [num_t_sum])

            ########################################################
            ################## (2) Control R #######################
            ########################################################

            if params['RUINER_MODE']=='use-r':
                if Wasserstein_D[0] > params['TARGET_W_DISTANCE']:
                    update_type = 'g'
                else:
                    update_type = 'r'
            elif params['RUINER_MODE']=='none-r':
                update_type = 'g'
            elif params['RUINER_MODE']=='test-r':
                update_type = 'r'
            else:
                raise Exception('Unsupport')

        if stabling_mse:
            update_type = 'g'

        ########################################################
        ############# (3) Update G network or R ################
        ########################################################

        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
        prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

        netG.zero_grad()

        G_cost = [0.0]
        R_cost = [0.0]
        if update_type=='g':

            for p in netD.parameters():
                p.requires_grad = False

            noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
            prediction_v = netG(
                noise_v = autograd.Variable(noise),
                state_v = autograd.Variable(state)
            ).narrow(1,params['STATE_DEPTH'],1)

            if stabling_mse:
                # print(state)
                # print(prediction_gt)
                # print(iiii)
                S = mse_loss_model(prediction_v, autograd.Variable(prediction_gt))
                S.backward()
                S_cost = S.data.cpu().numpy()
                logger.plot('S_cost', S_cost)
                G_R = 'S'
                logger.plot('G_R', np.asarray([0.0]))

            else:

                G = netD(
                        state_v = autograd.Variable(state),
                        prediction_v = prediction_v
                    ).mean()
                G.backward(mone)
                G_cost = -G.data.cpu().numpy()
                logger.plot('G_cost', G_cost)
                G_R = 'G'
                logger.plot('G_R', np.asarray([1.0]))

        elif update_type=='r':
            noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
            prediction_gt_r_v = netG(
                noise_v = autograd.Variable(noise),
                state_v = autograd.Variable(prediction_gt)
            ).narrow(1,(params['STATE_DEPTH']-1),1)

            R = mse_loss_model(prediction_gt_r_v, autograd.Variable(prediction_gt))

            R.backward()
            R_cost = R.data.cpu().numpy()
            logger.plot('R_cost', R_cost)
            G_R = 'R'
            logger.plot('G_R', np.asarray([-1.0]))
        
        optimizerG.step()

        ############################
        # (4) Log summary
        ############################

        if stabling_mse:
            print('[{}][{:<6}] L1: {:2.4f} AC: {:2.4f} S_cost:{:2.4f}'
                .format(
                    MULTI_RUN,
                    iteration,
                    L1,
                    AC,
                    S_cost[0],
                )
            )
        else:
            print('[{}][{:<6}] L1: {:2.4f} AC: {:2.4f} T: {:2.4f} W_cost:{:2.4f} GP_cost:{:2.4f} D_cost:{:2.4f} G_cost:{:2.4f}'
                .format(
                    MULTI_RUN,
                    iteration,
                    L1,
                    AC,
                    num_t_sum,
                    Wasserstein_D[0],
                    GP_cost[0],
                    D_cost[0],
                    G_cost[0],
                )
            )

        if iteration % LOG_INTER == 5:
            torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
            torch.save(netC.state_dict(), '{0}/netC.pth'.format(LOGDIR))
            torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))

            if LOG_INTER==TrainTo:
                if iteration>=TrainTo:
                    L1, AC = generate_image(iteration)
            else:
                L1, AC = generate_image(iteration)

            if stabling_mse:
                _, y = logger.get_plot('S_cost')
                stable_range = 200
                try:
                    stable = np.abs(np.mean(y[-1-stable_range*2:-1-stable_range])-np.mean(y[-1-stable_range:-1]))
                    print('Stable to:'+str(stable))
                    if stable<params['STABLE_MSE']:
                        stabling_mse = False
                except Exception as e:
                    pass
                

    if iteration % LOG_INTER == 5:
        logger.flush()

    logger.tick()

    
