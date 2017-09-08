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
import imageio
from decision_tree import *

CLEAR_RUN = False # if delete logdir and start a new run
MULTI_RUN = 'noise_encourage_marble' # display a tag before the result printed
GPU = "0" # use which GPU

MULTI_RUN = MULTI_RUN + '|GPU:' + GPU # this is a lable displayed before each print and log, to identify different runs at the same time on one computer
os.environ["CUDA_VISIBLE_DEVICES"] = GPU # set env variable that make the GPU you select
# after mask GPU, import torch
if GPU!=None:
    import torch
    import torch.autograd as autograd
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    torch.manual_seed(4213)
    GPU = range(torch.cuda.device_count()) # use all GPU you select
    print('Using GPU:'+str(GPU))

params = {}
params_seq = []
def add_parameters(**kwargs):
    global params_seq
    params_seq += kwargs.keys()
    params.update(kwargs)

'''domain settings'''
add_parameters(EXP = 'noise_encourage_exp') # the first level of log dir
add_parameters(DOMAIN = 'marble') # 1Dflip, 1Dgrid, 2Dgrid, marble
add_parameters(FIX_STATE = False) # whether to fix the start state at a specific point, this will simplify training. Usually using it for debugging so that you can have a quick run.
add_parameters(REPRESENTATION = chris_domain.IMAGE) # chris_domain.SCALAR, chris_domain.VECTOR, chris_domain.IMAGE
add_parameters(GRID_SIZE = 5) # size of 1Dgrid, 1Dflip, 2Dgrid

'''
domain dynamic
'''
if params['DOMAIN']=='1Dflip':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/params['GRID_SIZE']]*params['GRID_SIZE'])
    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.5]*2+[0.0]*(params['GRID_SIZE']-2))

elif params['DOMAIN']=='1Dgrid':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/3.0,2.0/3.0])

elif params['DOMAIN']=='2Dgrid':
    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.5,0.5,0.0,0.0])
    # add_parameters(OBSTACLE_POS_LIST = [])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.8, 0.0, 0.1, 0.1])
    # add_parameters(OBSTACLE_POS_LIST = [])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    # add_parameters(OBSTACLE_POS_LIST = [])

    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.8, 0.0, 0.1, 0.1])
    # add_parameters(OBSTACLE_POS_LIST = [(2, 2)])

    add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    add_parameters(OBSTACLE_POS_LIST = [(2, 2)])

    add_parameters(RANDOM_BACKGROUND = False)

    if params['RANDOM_BACKGROUND']==True:
        add_parameters(FEATURE = 1)
    else:
        add_parameters(FEATURE = 1)

elif params['DOMAIN']=='marble':
    add_parameters(FEATURE = 1)
    add_parameters(IMAGE_SIZE = 64)
    add_parameters(FRAME_INTERVAL = 3)

else:
    raise Exception('unsupport')

'''
method settings
'''
add_parameters(METHOD = 's-gan') # tabular, bayes-net-learner, deterministic-deep-net, s-gan

add_parameters(GP_MODE = 'pure-guide') # none-guide, use-guide, pure-guide
# add_parameters(GP_MODE = 'none-guide') # none-guide, use-guide, pure-guide
add_parameters(GP_GUIDE_FACTOR = 1.0)

add_parameters(INTERPOLATES_MODE = 'auto') # auto, one
# add_parameters(INTERPOLATES_MODE = 'one') # auto, one

# add_parameters(NOISE_ENCOURAGE = False)
add_parameters(NOISE_ENCOURAGE = True)

if params['DOMAIN']=='marble':
    add_parameters(NOISE_ENCOURAGE_FACTOR = 0.1)
else:
    add_parameters(NOISE_ENCOURAGE_FACTOR = 1.0)

'''
compute delta for differant domains
'''
BASE = 0.1 / ( ( (1)**0.5 ) / ( (5)**0.5 ) )
if params['DOMAIN']=='1Dflip' or params['DOMAIN']=='1Dgrid':

    if params['REPRESENTATION']==chris_domain.VECTOR:
        add_parameters(
            DELTA_T = ( BASE * ( ( (1)**0.5 ) / ( (params['GRID_SIZE'])**0.5 ) ) )
        )

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        add_parameters(
            DELTA_T = ( BASE * ( ( (chris_domain.BLOCK_SIZE**2)**0.5 ) / ( ( (chris_domain.BLOCK_SIZE**2)*params['GRID_SIZE'])**0.5 ) ) )
        )

    else:
        raise Exception('s')

elif params['DOMAIN']=='2Dgrid':

    if params['REPRESENTATION']==chris_domain.VECTOR:
        add_parameters(
            DELTA_T = ( BASE * ( ( (1**2)**0.5 ) / ( (params['GRID_SIZE']**2)**0.5 ) ) )
        )

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        if params['RANDOM_BACKGROUND']==False:
            add_parameters(
                DELTA_T = ( BASE * ( ( (chris_domain.BLOCK_SIZE**2)**0.5 ) / ( ( ( (chris_domain.BLOCK_SIZE*params['GRID_SIZE'])**2)*1)**0.5 ) ) )
            )
        else:
            add_parameters(
                DELTA_T = ( BASE * ( ( (chris_domain.BLOCK_SIZE**2)**0.5 ) / ( ( ( (chris_domain.BLOCK_SIZE*params['GRID_SIZE'])**2)*2)**0.5 ) ) )
            )

    else:
        raise Exception('s')

elif params['DOMAIN']=='marble':
    add_parameters(
        DELTA_T = 2.0*( BASE * ( ( ( (0.5*params['IMAGE_SIZE']*2.4/14.9)**2)**0.5 ) / ( ( ( (params['IMAGE_SIZE'])**2)*params['FEATURE'])**0.5 ) ) )
    )

else:
    raise Exception('s')

'''model settings'''
add_parameters(DIM = 128) # warnning: this is not likely to make a difference, but the result I report except the random bg domain is on DIM = 512
add_parameters(NOISE_SIZE = 8) # warnning: this is not likely to make a difference, but the result I report except the random bg domain is on NOISE_SIZE = 128, when using noise reward, we can set this to be smaller
add_parameters(BATCH_SIZE = 32)
add_parameters(DATASET_SIZE = 33554) # 1610612736 # warnning: this is not likely to make a difference, but the result I report except the random bg domain is on dynamic full data set
# LAMBDA is set seperatly for different representations
if params['REPRESENTATION']==chris_domain.SCALAR:
    add_parameters(LAMBDA = 0.1)

elif params['REPRESENTATION']==chris_domain.VECTOR:
    add_parameters(LAMBDA = 5)

elif params['REPRESENTATION']==chris_domain.IMAGE:
    add_parameters(LAMBDA = 10)

else:
    raise Exception('Unsupport')

'''
marble domain needs different STATE_DEPTH
'''
if params['DOMAIN']=='marble':
    add_parameters(STATE_DEPTH = 2)

else:
    add_parameters(STATE_DEPTH = 1)

'''
build domains according to the settings
'''
if params['DOMAIN']=='1Dflip':
    domain = chris_domain.BitFlip1D(
        length=params['GRID_SIZE'],
        mode=params['REPRESENTATION'],
        prob_dirs=params['GRID_ACTION_DISTRIBUTION'],
        fix_state=params['FIX_STATE'],
        soft_vector=params['SOFT_VECTOR']
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
        should_wrap=False,
        fix_state=params['FIX_STATE'],
        random_background = params['RANDOM_BACKGROUND'],
    )

elif params['DOMAIN']=='marble':
    pass

else:
    print(unsupport)

add_parameters(AUX_INFO = '')

'''
summary settings
'''
DSP = ''
params_str = 'Settings'+'\n'
params_str += '##################################'+'\n'
for i in range(len(params_seq)):
    DSP += params_seq[i]+'_'+str(params[params_seq[i]]).replace('.','_').replace(',','_').replace(' ','_')+'/'
    params_str += params_seq[i]+' >> '+str(params[params_seq[i]])+'\n'
params_str += '##################################'+'\n'
print(params_str)

'''
build log dir
'''
BASIC = '../../result/'
LOGDIR = BASIC+DSP
if CLEAR_RUN:
    subprocess.call(["rm", "-r", LOGDIR])
subprocess.call(["mkdir", "-p", LOGDIR])
with open(LOGDIR+"Settings.txt","a") as f:
    f.write(params_str)

N_POINTS = 128 # for scalar domain, data tendity when draw the critic surface
RESULT_SAMPLE_NUM = 1000 # number of samples to draw when evaluate L1 loss
TrainTo   = 100000 # train to 100k and evaluate

LOG_INTER = 1000
if params['DOMAIN']=='marble':
    LOG_INTER = 500 # marble is slower, log more

'''
set DESCRIBE_DIM for low dimensional domain
'''
if params['REPRESENTATION']==chris_domain.SCALAR:

    if params['DOMAIN']=='2Dgrid':
        DESCRIBE_DIM = 2

    else:
        raise Exception('s')

elif params['REPRESENTATION']==chris_domain.VECTOR:

    if params['DOMAIN']=='1Dgrid' or params['DOMAIN']=='1Dflip':
        DESCRIBE_DIM = params['GRID_SIZE']

    else:
        raise Exception('s')

if params['DOMAIN']=='marble':
    PRE_DATASET = False
    LOG_SEQ_NUM = 32
    LOG_SEQ_LENTH = 8
    
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

def log_img(x,name,iteration=0,nrow=8):
    if params['REPRESENTATION']==chris_domain.VECTOR:
        x = vector2image(x)
    x = x.squeeze(1)
    if params['DOMAIN']=='2Dgrid':
        if x.size()[1]==2:
            log_img_final(x[:,0:1,:,:],name+'_b',iteration,nrow)
            log_img_final(x[:,1:2,:,:],name+'_a',iteration,nrow)
            x = torch.cat([x,x[:,0:1,:,:]],1)
    log_img_final(x,name,iteration,nrow)

def log_img_final(x,name,iteration=0,nrow=8):
    vutils.save_image(x, LOGDIR+name+'_'+str(iteration)+'.png')
    vis.images( 
        x.cpu().numpy(),
        win=str(MULTI_RUN)+'-'+name,
        opts=dict(caption=str(MULTI_RUN)+'-'+name+'_'+str(iteration)),
        nrow=nrow,
    )

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
            
            '''
            low dimensional domain share following network
            '''
            conv_layer = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.001, inplace=True),
            )

            deconv_layer = nn.Sequential(
                nn.Linear(params['DIM'], DESCRIBE_DIM),
                nn.Sigmoid()
            )

        elif params['REPRESENTATION']==chris_domain.IMAGE:

            if params['DOMAIN']!='marble':

                '''
                image domains that are not marble share following network6
                '''
                conv_layer = nn.Sequential(
                    nn.Conv3d(
                        in_channels=params['FEATURE'],
                        out_channels=64,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    nn.Conv3d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                )

                '''
                compute the number of the last conv layer
                '''
                if params['DOMAIN']=='1Dgrid':
                    temp = 128*1*(params['GRID_SIZE'])
                elif params['DOMAIN']=='2Dgrid':
                    temp = 128*1*(params['GRID_SIZE']**2)
                else:
                    raise Exception('s')

                squeeze_layer = nn.Sequential(
                    nn.Linear(temp, params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                cat_layer = nn.Sequential(
                    nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                unsqueeze_layer = nn.Sequential(
                    nn.Linear(params['DIM'], temp),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                deconv_layer = nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    nn.ConvTranspose3d(
                        in_channels=64,
                        out_channels=params['FEATURE'],
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.Sigmoid()
                )

            elif params['DOMAIN']=='marble':

                '''
                marble domain use a another network
                '''
                conv_layer = nn.Sequential(
                    # params['FEATURE']*2*64*64
                    nn.Conv3d(
                        in_channels=1,
                        out_channels=64,
                        kernel_size=(2,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 64*1*32*32
                    nn.Conv3d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 128*1*16*16
                    nn.Conv3d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 256*1*8*8
                )
                temp = 256*1*8*8
                squeeze_layer = nn.Sequential(
                    nn.Linear(temp, params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                cat_layer = nn.Sequential(
                    nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                unsqueeze_layer = nn.Sequential(
                    nn.Linear(params['DIM'], temp),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                deconv_layer = nn.Sequential(
                    # 256*1*8*8
                    nn.ConvTranspose3d(
                        in_channels=256,
                        out_channels=128,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 128*1*16*16
                    nn.ConvTranspose3d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 64*1*32*32
                    nn.ConvTranspose3d(
                        in_channels=64,
                        out_channels=1,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.Sigmoid()
                    # 1*1*64*64
                )

        else:
            raise Exception('representation unsupport!')

        self.conv_layer = conv_layer
        self.squeeze_layer = squeeze_layer
        self.cat_layer = cat_layer
        self.unsqueeze_layer = unsqueeze_layer
        self.deconv_layer = deconv_layer
        
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
            '''flatten'''
            temp = x.size()
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        cat_input = x
        x = self.cat_layer(torch.cat([x,noise_v],1))
        x = self.unsqueeze_layer(x)
        if params['REPRESENTATION']==chris_domain.IMAGE:
            '''transpose flatten'''
            x = x.view(temp)
        defore_deconv = x
        x = self.deconv_layer(x)

        '''decompose'''
        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:
            x = x.unsqueeze(1)

        elif params['REPRESENTATION']==chris_domain.IMAGE:
            # N*F*D*H*W to N*D*F*H*W
            x = x.permute(0,2,1,3,4)

        return x, defore_deconv, cat_input

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        if params['REPRESENTATION']==chris_domain.SCALAR or params['REPRESENTATION']==chris_domain.VECTOR:

            '''
            low dimensional domain share following network
            there are 'inplace=True' here, it works when compute grandien of the gradient
            '''
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

            if params['DOMAIN']!='marble':
            
                '''
                image domains that are not marble share following network
                '''
                conv_layer = nn.Sequential(
                    nn.Conv3d(
                        in_channels=params['FEATURE'],
                        out_channels=64,
                        kernel_size=(2,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    nn.Conv3d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                )

                '''
                compute the number of the last conv layer
                '''
                if params['DOMAIN']=='1Dgrid':
                    temp = 128*1*(params['GRID_SIZE'])
                elif params['DOMAIN']=='2Dgrid':
                    temp = 128*1*(params['GRID_SIZE']**2)
                else:
                    raise Exception('s')

                squeeze_layer = nn.Sequential(
                    nn.Linear(temp, params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                final_layer = nn.Sequential(
                    nn.Linear(params['DIM'], params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                    nn.Linear(params['DIM'], 1),
                )

            elif params['DOMAIN']=='marble':

                '''
                marble domain use a another network
                '''
                conv_layer = nn.Sequential(
                    # 1*3*64*64
                    nn.Conv3d(
                        in_channels=1,
                        out_channels=64,
                        kernel_size=(2,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 64*2*32*32
                    nn.Conv3d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(2,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 128*1*16*16
                    nn.Conv3d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(1,4,4),
                        stride=(1,2,2),
                        padding=(0,1,1),
                        bias=False,
                    ),
                    nn.LeakyReLU(0.001, inplace=True),
                    # 256*1*8*8
                )
                temp = 256*1*8*8
                squeeze_layer = nn.Sequential(
                    nn.Linear(temp, params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                )
                final_layer = nn.Sequential(
                    nn.Linear(params['DIM'], params['DIM']),
                    nn.LeakyReLU(0.001, inplace=True),
                    nn.Linear(params['DIM'], 1),
                )

        else:
            raise Exception('Unsupport')

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
                '''flatten'''
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
        x = torch.from_numpy(np.array(x)).cuda().unsqueeze(1).permute(0,1,4,2,3).float()

    else:
        raise Exception('ss')
    
    return x

def song2chris(x):

    if (params['REPRESENTATION']==chris_domain.SCALAR) or (params['REPRESENTATION']==chris_domain.VECTOR):
        x = x.squeeze(1).cpu().numpy()

    elif params['REPRESENTATION']==chris_domain.IMAGE:
        x = x.permute(0,1,3,4,2).squeeze(1).cpu().numpy()

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

    if params['DOMAIN']!='marble':

        domain.reset()
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
                    ).data

                elif params['METHOD']=='s-gan':
                    '''prediction'''
                    noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
                    prediction = netG(
                        noise_v = autograd.Variable(noise, volatile=True),
                        state_v = autograd.Variable(start_state_batch, volatile=True)
                    )[0].data

                else:
                    raise Exception('Unsupport')

            else:
                
                prediction = get_tabular_samples(tabular,start_state[0].cpu().numpy())
                
                if prediction is not None:
                    prediction = torch.cuda.FloatTensor(prediction)

            log_img(start_state_batch,'state_'+str(ii),iteration)
            log_img(prediction,'prediction_'+str(ii),iteration)

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


        l1 = np.mean(all_l1)
        ac = np.mean(all_ac)

    elif params['DOMAIN']=='marble':
        state = domain.get_batch().narrow(1,0,params['STATE_DEPTH'])
        state = state[0:1]
        state = torch.cat([state]*LOG_SEQ_NUM,0)

        seq = state

        for t in range(LOG_SEQ_LENTH):
            '''prediction'''
            if params['METHOD']=='s-gan':
                noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                prediction = netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(state, volatile=True)
                )[0].data
            elif params['METHOD']=='deterministic-deep-net':
                prediction = netT(
                    state_v = autograd.Variable(state, volatile=True)
                ).data

            seq = torch.cat([seq,prediction],1)

            state = seq.narrow(1,seq.size()[1]-params['STATE_DEPTH'],params['STATE_DEPTH'])

        seq = seq.contiguous().view(-1,1,seq.size()[2],seq.size()[3],seq.size()[4])

        log_img(
            seq,
            'seq',
            iteration,
            nrow=(LOG_SEQ_LENTH+params['STATE_DEPTH']),
        )

        l1, ac = 0.0, 0.0

    return l1, ac

def evaluate_domain(iteration,tabular=None):

    if params['REPRESENTATION']==chris_domain.SCALAR:

        batch_size = (N_POINTS**2)

        domain.set_fix_state(True)
        data_fix_state = dataset_iter(
            fix_state=True,
            batch_size=batch_size
        )
        dataset = data_fix_state.next()
        domain.set_fix_state(params['FIX_STATE'])

        evaluate_domain_with_filter(
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

def evaluate_domain_with_filter(iteration,dataset,gen_basic=False,filter_net=None):

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

    '''prediction'''
    noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
    prediction = netG(
        noise_v = autograd.Variable(noise, volatile=True),
        state_v = autograd.Variable(state, volatile=True)
    )[0].data
    
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

class marble_domain(object):
    """docstring for marble_domain"""
    def __init__(self):
        super(marble_domain, self).__init__()

        self.indexs_selector = torch.LongTensor(params['BATCH_SIZE'])

        file_list = ['00014','00015','00016','00017','00018','00019','00020']

        if PRE_DATASET:

            logger = lib.plot.logger(LOGDIR,DSP,params_str,MULTI_RUN)

            file = file_list[6]
            file_name = '../../dataset/marble/single/'+file

            print('creating marble dataset from MTS')

            vid = imageio.get_reader(file_name+'.MTS', 'ffmpeg')
            info = vid.get_meta_data()
            print(info)

            def get_frame(processed_frame):
                try:
                    image = vid.get_data(processed_frame)
                except Exception as e:
                    return None
                image = image[:,:,1]
                image = cv2.resize(image,(params['IMAGE_SIZE'],params['IMAGE_SIZE']))
                image = torch.from_numpy(image).int()
                image = image.unsqueeze(0)
                image = image.unsqueeze(0)
                return image

            mask = torch.IntTensor(get_frame(0).size()).fill_(1)
            mask[0,0,0:int(params['IMAGE_SIZE']*3.2/8.4),0:int(params['IMAGE_SIZE']*2.0/8.4)].fill_(0)
            mask[0,0,int(params['IMAGE_SIZE']-params['IMAGE_SIZE']*3.2/8.4):params['IMAGE_SIZE'],0:int(params['IMAGE_SIZE']*2.1/8.4)].fill_(0)

            fram_interval = params['FRAME_INTERVAL']

            frame_start = 0
            while True:
                frame_start += 1

                delta = 0.0
                data = None
                image = None
                last_image = None
                breaking = False
                processed_frame_dic = []
                for frame_i in range(params['STATE_DEPTH']+1):

                    processed_frame = frame_start+frame_i*fram_interval
                    processed_frame_dic += [processed_frame]

                    image = get_frame(processed_frame)

                    if image is None:
                        breaking = True
                        break
                    
                    if last_image is not None:
                        this_delta = (image*mask-last_image*mask).abs().sum()
                        delta += this_delta
                    else:
                        last_image = image

                    try:
                        data = torch.cat([data,image],0)
                    except Exception as e:
                        data = image

                if breaking:
                    break

                logger.plot('delta_'+file, delta)
                logger.tick()
                
                accept = False
                if (delta > 20000) and (delta < 70000):

                    accept = True

                    logger.flush()

                    vis.images(
                        data.cpu().numpy(),
                    )

                    data = data.unsqueeze(0)

                    try:
                        self.dataset = torch.cat([self.dataset,data],0)
                    except Exception as e:
                        self.dataset = data
                
                try:
                    print('[{:2.4f}%]Get data from {} at [{}/{}] with delta: {}. Accept: {}. Dataset: {}'
                        .format(
                            (float(frame_start)/info['nframes']*100.0),
                            file,
                            processed_frame_dic,
                            info['nframes'],
                            delta,
                            accept,
                            self.dataset.size()
                        )
                    )
                except Exception as e:
                    pass
                
            print('Save marble dateset '+str(self.dataset.size())+' to npz')
            np.save(file_name, self.dataset.cpu().numpy())

            raise Exception('Creat dataset done.')

        else:

            for file in file_list:

                file_name = '../../dataset/marble/single/'+file

                try:
                    data = torch.from_numpy(np.load(file_name+'.npy'))
                    print('Load data from '+file+' : '+str(data.size()))
                    try:
                        self.dataset = torch.cat([self.dataset,data],0)
                    except Exception as e:
                        self.dataset = data
                except Exception as e:
                    print('Failed to load data from '+file)

            self.dataset = self.dataset.float()/255.0
            self.dataset = self.dataset.cuda()

            print('Got marble dateset: '+str(self.dataset.size()))

            log_batch = self.get_batch()
            for b in range(log_batch.size()[0]):
                vis.images(
                    log_batch[b].cpu().numpy(),
                )

    def get_batch(self):
        indexs = self.indexs_selector.random_(0,self.dataset.size()[0]).cuda()
        return torch.index_select(self.dataset,0,indexs)

class grid_domain(object):
    """docstring for grid_domain"""
    def __init__(self):
        super(grid_domain, self).__init__()

        self.dataset_lenth = params['DATASET_SIZE']

        self.indexs_selector = torch.LongTensor(params['BATCH_SIZE'])

        # file = '5x5_random_bg_3_small10'
        file = '5x5_nbg_ob'
        file_name = '../../dataset/grid/'+file

        try:
            self.dataset = torch.from_numpy(np.load(file_name+'.npy')).cuda()
            print('Load dataset from '+file+' : '+str(self.dataset.size()))

        except Exception as e:

            print('Failed to load dataset from '+file)
            print('Creating dataset')

            while True:

                domain.reset()
                ob = torch.from_numpy(domain.get_state()).unsqueeze(0)
                ob_next = torch.from_numpy(domain.update()).unsqueeze(0)

                data = torch.cat([ob,ob_next],0).unsqueeze(0)
                
                try:
                    self.dataset = torch.cat([self.dataset,data],0)
                except Exception as e:
                    self.dataset = data

                print('[{:2.4f}%]'
                    .format(
                        float(self.dataset.size()[0])/float(self.dataset_lenth)*100.0,
                    )
                )

                if self.dataset.size()[0]>self.dataset_lenth:
                    break

            if params['REPRESENTATION']==chris_domain.SCALAR:
                self.dataset = self.dataset.float()
            if params['REPRESENTATION']==chris_domain.VECTOR:
                self.dataset = self.dataset.float()
            elif params['REPRESENTATION']==chris_domain.IMAGE:
                self.dataset = self.dataset.permute(0,1,4,2,3)
                self.dataset = self.dataset.float()

            self.dataset = self.dataset.cuda()

            print('Got grid dateset: '+str(self.dataset.size()))

            for b in range(params['BATCH_SIZE']):
                vis.images(
                    self.dataset[b].cpu().numpy(),
                )

            print('Save marble dateset '+str(self.dataset.size())+' to npz')
            np.save(file_name, self.dataset.cpu().numpy())

    def get_batch(self):
        indexs = self.indexs_selector.random_(0,self.dataset.size()[0]).cuda()
        return torch.index_select(self.dataset,0,indexs)

'''
wrap domain is to store dataset
and generate dataset at the begining,
'''
if params['DOMAIN']=='marble':
    domain = marble_domain()
    wrap_domain = domain
else:
    wrap_domain = grid_domain()

def dataset_iter(fix_state=False, batch_size=params['BATCH_SIZE']):

    while True:

        dataset = wrap_domain.get_batch()

        # print(dataset.size())
        # print(dataset[3,0,0,:,:])
        # # # print(dataset[3,0,1,:,:])
        # print(dataset[3,1,0,:,:])
        # print(dataset[3,1,1,:,:])
        # # print(dataset[4,0,0,:,:])
        # # print(dataset[4,1,0,:,:])
        # # # # print('---')
        # # # # print(dataset[3,1,0,:,:])
        # # # # print(dataset[3,1,1,:,:])
        # # # # print(dataset[3,0,:])
        # # # # print(dataset[3,0,:])
        # # # print(dataset[3:7])
        # print(s)

        yield dataset

def calc_gradient_penalty(netD, state, prediction, prediction_gt, log=False):
    
    '''get multiple interplots'''
    if params['INTERPOLATES_MODE']=='auto':
        prediction_fl = prediction.contiguous().view(prediction.size()[0],-1)
        prediction_gt_fl = prediction_gt.contiguous().view(prediction_gt.size()[0],-1)
        max_norm = (prediction_gt_fl.size()[1])**0.5      
        d_mean = (prediction_gt_fl-prediction_fl).norm(2,dim=1)/max_norm
        # print(d_mean)
        # print(prediction_gt[0:4,0,1,:,:])
        # print(s)
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
        num_t_sum = params['BATCH_SIZE']
        alpha = torch.rand(prediction_gt.size()[0]).cuda()

    # if params['SOFT_GP']:
    #     '''solf function here'''
    #     alpha = (alpha*params['SOFT_GP_FACTOR']).tanh()
        
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

    # print(prediction_gt[0:1,0,1,:,:])
    # print(prediction[0:1,0,1,:,:])
    # print(interpolates[0:1,0,1,:,:])

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

    if params['GP_MODE']=='pure-guide':

        prediction_fl = prediction.contiguous().view(prediction.size()[0],-1)
        prediction_gt_fl = prediction_gt.contiguous().view(prediction_gt.size()[0],-1)

        gradients_direction_gt_fl = prediction_gt_fl - prediction_fl

        def torch_remove_at_batch(x,index):

            if x.size()[0]==1:
                return None

            if index==0:
                x = x[index+1:x.size()[0]]
            elif index==(x.size()[0]-1):
                x[0:index]
            else:
                x = torch.cat(
                    [x[0:index],x[index+1:x.size()[0]]],
                    0
                )

            return x

        original_size = gradients_direction_gt_fl.size()[0]

        b = 0
        while True:
            if gradients_direction_gt_fl[b].abs().max() < 0.01:
                gradients_direction_gt_fl = torch_remove_at_batch(
                    gradients_direction_gt_fl,
                    b
                )
                gradients_fl = torch_remove_at_batch(
                    gradients_fl,
                    b
                )
                if gradients_fl is None:
                    print('No valid batch, return')
                    return None, 0
            else:
                b += 1
                if b>=gradients_direction_gt_fl.size()[0]:
                    # print('Filter batch to: ' + str(gradients_direction_gt_fl.size()[0]))
                    break

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

def calc_gradient_reward(noise_v, prediction_v_before_deconv):

    def get_grad_norm(inputs,outputs):

        gradients = autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones(outputs.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.contiguous()
        gradients_fl = gradients.view(gradients.size()[0],-1)
        gradients_norm = gradients_fl.norm(2, dim=1) / ((gradients_fl.size()[1])**0.5)

        return gradients_norm

    gradients_norm_noise = get_grad_norm(noise_v,prediction_v_before_deconv)

    logger.plot('gradients_norm_noise', [gradients_norm_noise.data.mean()])

    gradients_reward = (gradients_norm_noise+1.0).log().mean()*params['NOISE_ENCOURAGE_FACTOR']

    return gradients_reward

def restore_model():
    print('Trying load models....')
    try:
        netD.load_state_dict(torch.load('{0}/netD.pth'.format(LOGDIR)))
        print('Previous checkpoint for netD founded')
    except Exception, e:
        print('Previous checkpoint for netD unfounded')
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

elif params['METHOD']=='bayes-net-learner':

    dataset = data("")
    data = dataset_iter()
    for i in range(100):
        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(
            dimension=1,
            start=0,
            length=params['STATE_DEPTH']
        ).squeeze().cpu().numpy()
        prediction_gt = state_prediction_gt.narrow(
            dimension=1,
            start=params['STATE_DEPTH'],
            length=1).squeeze().cpu().numpy()
        if i==0:
            for attr in range(np.shape(state)[1]):
                dataset.attributes += ['state_'+str(attr)]
                dataset.attr_types += ['true']
            for attr in range(np.shape(prediction_gt)[1]):
                dataset.attributes += ['prediction_'+str(attr)]
                dataset.attr_types += ['true']
        for b in range(np.shape(state)[0]):
            dataset.examples += [np.concatenate((state[b],prediction_gt[b]),0)]

    classifier = 'prediction_0'
    dataset.classifier = classifier
    #find index of classifier
    for a in range(len(dataset.attributes)):
        if dataset.attributes[a] == dataset.classifier:
            dataset.class_index = a
    if dataset.class_index is None:
        raise Exception('s')

    root = compute_tree(dataset, None, classifier)
    print(root.attr_split_value)

    print(s)

elif params['METHOD']=='deterministic-deep-net':
    netT = Generator().cuda()
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

elif params['METHOD']=='s-gan':

    '''build models'''
    netG = Generator().cuda()
    netD = Discriminator().cuda()
    print netG
    print netD

    '''init models'''
    netD.apply(weights_init)
    netG.apply(weights_init)

    '''optimizers'''
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.0, 0.9))

    mse_loss_model = torch.nn.MSELoss(size_average=True)

    one = torch.FloatTensor([1]).cuda()
    mone = one * -1

    '''try restore model'''
    restore_model()

    L1, AC = 2.0, 0.0

'''build dataset iter'''
data = dataset_iter()

'''build and try restore logger'''
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
            l1,_ = evaluate_domain(iteration,tabular_dic)

        print('[{}][{}] l1: {}'
            .format(
                MULTI_RUN,
                iteration,
                l1
            )
        )

    elif params['METHOD']=='bayes-net-learner':
        pass

    elif params['METHOD']=='deterministic-deep-net':

        '''
            get data set
            format:
                # 0 dimension: batch size
                # 1 dimension: depth (time), t_0 (state), t (prediction_gt)
                # 2 dimension: feature (channel), for xianming default to 1
                # 3 & 4 dimension: size (1D or 2D)
        '''
        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(
            dimension=1,
            start=0,
            length=params['STATE_DEPTH']
        )
        prediction_gt = state_prediction_gt.narrow(
            dimension=1,
            start=params['STATE_DEPTH'],
            length=1)

        netT.zero_grad()
        
        prediction = netT(
            state_v = autograd.Variable(state)
        )

        T = mse_loss_model(prediction, autograd.Variable(prediction_gt))

        T.backward()

        T_cost = T.data.cpu().numpy()
        logger.plot('T_cost', T_cost)
        
        optimizerT.step()

        if iteration % LOG_INTER == 5:
            # evaluate the L1 loss and accept percentage
            L1, AC = evaluate_domain(iteration)

        print('[{}][{:<6}] T_cost:{:2.4f} L1::{:2.4f} AC::{:2.4f}'
            .format(
                MULTI_RUN,
                iteration,
                T_cost[0],
                L1,
                AC
            )
        )

    elif params['METHOD']=='s-gan':

        ########################################################
        ############### (1) Update D network ###################
        ########################################################

        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in xrange(5):

            '''get data set'''
            state_prediction_gt = data.next()
            state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
            prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

            '''get generated data'''
            noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
            prediction = netG(
                noise_v = autograd.Variable(noise, volatile=True),
                state_v = autograd.Variable(state, volatile=True)
            )[0].data

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

            D_cost = D_fake - D_real + gradient_penalty

            D_cost = D_cost.data.cpu().numpy()

            Wasserstein_D = (D_real - D_fake).data.cpu().numpy()

            optimizerD.step()

        logger.plot('GP_cost', GP_cost)
        logger.plot('D_cost', D_cost)
        logger.plot('W_dis', Wasserstein_D)
        logger.plot('num_t_sum', [num_t_sum])

        ########################################################
        ############# (3) Update G network or R ################
        ########################################################

        state_prediction_gt = data.next()
        state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
        prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

        netG.zero_grad()

        for p in netD.parameters():
            p.requires_grad = False

        noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()

        noise_v = autograd.Variable(
            noise,
            requires_grad=True
        )
        prediction_v, prediction_v_before_deconv, _ = netG(
            noise_v = noise_v,
            state_v = autograd.Variable(state)
        )
        G = netD(
            state_v = autograd.Variable(state),
            prediction_v = prediction_v
        ).mean()
        G.backward(
            mone,
            retain_graph=params['NOISE_ENCOURAGE'],
        )
        G_cost = -G.data.cpu().numpy()
        logger.plot('G_cost', G_cost)
        
        GR_cost = [0.0]
        if params['NOISE_ENCOURAGE']:
            gradients_reward = calc_gradient_reward(
                noise_v=noise_v,
                prediction_v_before_deconv=prediction_v_before_deconv,
            )
            gradients_reward.backward(mone)
        GR_cost = gradients_reward.data.cpu().numpy()
        logger.plot('GR_cost', GR_cost)

        optimizerG.step()

        ############################
        # (4) Log summary
        ############################

        print('[{}][{:<6}] L1: {:2.4f} AC: {:2.4f} T: {:2.4f} W_cost:{:2.4f} GP_cost:{:2.4f} D_cost:{:2.4f} G_cost:{:2.4f} GR_cost:{:2.4f}'
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
                GR_cost[0],
            )
        )

        if iteration % LOG_INTER == 5:
            torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
            torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))

            if LOG_INTER==TrainTo:
                if iteration>=TrainTo:
                    L1, AC = evaluate_domain(iteration)
            else:
                L1, AC = evaluate_domain(iteration)

            if params['NOISE_ENCOURAGE']:
                if params['DOMAIN']=='marble':
                    if iteration>30*1000:
                        params['NOISE_ENCOURAGE'] = False
                else:
                    if L1<0.5:
                        params['NOISE_ENCOURAGE'] = False

    if iteration % LOG_INTER == 5:
        logger.flush()

    logger.tick()

    
