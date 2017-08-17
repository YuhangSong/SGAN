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

import subprocess
from PIL import Image
import torchvision.utils as vutils
import visdom
vis = visdom.Visdom()
import time

CUDA = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
if CUDA!=None:
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
'''main settings'''
add_parameters(EXP = 'exp_2_10')
add_parameters(DATASET = '1Dgrid') # 1Dgrid, 1Dflip, 2Dgrid,
add_parameters(GAME_MDOE = 'full') # same-start, full
add_parameters(DOMAIN = 'vector') # scalar, vector, image
add_parameters(METHOD = 'grl') # tabular, bayes-net-learner, deterministic-deep-net, grl
add_parameters(RUINER_MODE = 'use-r') # none-r, use-r, test-r
add_parameters(GRID_SIZE = 5)


'''default setting'''
add_parameters(GAN_MODE = 'wgan-grad-panish') # wgan, wgan-grad-panish, wgan-gravity, wgan-decade
add_parameters(FILTER_MODE = 'filter-d-c') # none-f, filter-c, filter-d, filter-d-c
add_parameters(CORRECTOR_MODE = 'c-decade') # c-normal, c-decade
add_parameters(OPTIMIZER = 'Adam') # Adam, RMSprop

'''some special settings if use-r'''
if params['RUINER_MODE']=='use-r':
    add_parameters(FASTEN_D = 10)
    add_parameters(GP_TO = 0.0)
    add_parameters(G_INIT_SIGMA = 0.00002)

else:
    add_parameters(FASTEN_D = 1)
    add_parameters(GP_TO = 1.0)
    add_parameters(G_INIT_SIGMA = 0.02)

add_parameters(GRID_BACKGROUND = 0.1)
add_parameters(GRID_FOREGROUND = 0.9)

if params['DATASET']=='1Dflip':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/params['GRID_SIZE']]*params['GRID_SIZE'])
    FIX_STATE_TO = [params['GRID_FOREGROUND']]*(params['GRID_SIZE']/2)+[params['GRID_BACKGROUND']]*(params['GRID_SIZE']/2+1)

elif params['DATASET']=='1Dgrid':
    add_parameters(GRID_ACTION_DISTRIBUTION = [1.0/3.0,2.0/3.0])
    FIX_STATE_TO = [params['GRID_SIZE']/2,0]

elif params['DATASET']=='2Dgrid':
    # add_parameters(GRID_ACTION_DISTRIBUTION = [0.8,0.1,0.0,0.1])
    add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    FIX_STATE_TO = [params['GRID_SIZE']/2,params['GRID_SIZE']/2]

else:
    print(unsupport)

if params['DOMAIN']=='scalar':
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = 2)
    add_parameters(LAMBDA = 0.1)
    add_parameters(BATCH_SIZE = 256)
    add_parameters(TARGET_W_DISTANCE = 0.1)

elif params['DOMAIN']=='vector':
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = params['GRID_SIZE'])
    add_parameters(LAMBDA = 1)
    add_parameters(BATCH_SIZE = 64)
    add_parameters(TARGET_W_DISTANCE = 0.1)

elif params['DOMAIN']=='image':
    add_parameters(DIM = 128)
    add_parameters(NOISE_SIZE = 128)
    add_parameters(LAMBDA = 10)
    add_parameters(BATCH_SIZE = 64)
    add_parameters(TARGET_W_DISTANCE = 0.1)

else:
    print(unsupport)
    
add_parameters(CRITIC_ITERS = 5)
add_parameters(GRID_DETECTION = 'threshold')
add_parameters(GRID_ACCEPT = 0.1)
add_parameters(MULTI_RUN = '0')

DSP = ''
params_str = 'Settings'+'\n'
params_str += '##################################'+'\n'
for i in range(len(params_seq)):
    DSP += params_seq[i]+'_'+str(params[params_seq[i]]).replace('.','_').replace(',','_').replace(' ','_')+'/'
    params_str += params_seq[i]+' >> '+str(params[params_seq[i]])+'\n'
params_str += '##################################'+'\n'
print(params_str)

############################### Generated Settings ###############################
BASIC = '../../result/'
LOGDIR = BASIC+DSP
subprocess.call(["mkdir", "-p", LOGDIR])
with open(LOGDIR+"Settins.txt","a") as f:
    f.write(params_str)

N_POINTS = 128
RESULT_SAMPLE_NUM = 2000
FILTER_RATE = 0.5
LOG_INTER = 100

if params['DOMAIN']=='scalar':
    if params['DATASET']=='2Dgrid':
        DESCRIBE_DIM = 2

    else:
        print(unsupport)

elif params['DOMAIN']=='vector':
    if params['DATASET']=='1Dgrid' or params['DATASET']=='1Dflip':
        DESCRIBE_DIM = params['GRID_SIZE']

    else:
        print(unsupport)

if params['DATASET']=='marble':
    add_parameters(STATE_DEPTH = 3)
    add_parameters(FEATURE = 3)
    add_parameters(IMAGE_SIZE = 128)

else:
    add_parameters(STATE_DEPTH = 1)
    add_parameters(FEATURE = 1)
    add_parameters(IMAGE_SIZE = 32)

BOX_SIZE = params['IMAGE_SIZE']/params['GRID_SIZE']

############################### Definition Start ###############################

def vector2image(x):
    x_temp = torch.FloatTensor(
        x.size()[0],
        x.size()[1],
        1,
        params['IMAGE_SIZE']/params['GRID_SIZE'],
        params['IMAGE_SIZE']).cuda().fill_(0.0)
    for b in range(x.size()[0]):
        for d in range(x.size()[1]):
            for i in range(x.size()[2]):
                from_ = i*params['IMAGE_SIZE']/params['GRID_SIZE']
                to_ = (i+1)*params['IMAGE_SIZE']/params['GRID_SIZE']
                fill_ = float(x[b][d][i])
                x_temp[b,d,0,:,from_:to_].fill_(fill_)
    return x_temp

def log_img(x,name,iteration):
    if params['DOMAIN']=='vector':
        x = vector2image(x)
    x = x.squeeze(1)
    vutils.save_image(x, LOGDIR+name+'_'+str(iteration)+'.png')
    vis.images( x.cpu().numpy(),
                win=CUDA+'-'+name,
                opts=dict(caption=CUDA+'-'+name+'_'+str(iteration)))

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
                win=CUDA+'-'+win,
                opts=dict(title=CUDA+'-'+name))

class D_out_layer(nn.Module):

    def __init__(self):
        super(D_out_layer, self).__init__()
    def forward(self, x):
        if params['GAN_MODE']=='wgan-gravity':
            for i in range(x.size()[0]):
                if x[i].data.cpu().numpy()[0]>0.0:
                    x[i] = torch.log(x[i]+1.0)
                elif x[i].data.cpu().numpy()[0]<0.0:
                    x[i] = -torch.log(-x[i]+1.0)
        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
            
            conv_layer = nn.Sequential(
                nn.Linear(DESCRIBE_DIM, params['DIM']),
                nn.BatchNorm1d(params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            squeeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.BatchNorm1d(params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                nn.BatchNorm1d(params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.BatchNorm1d(params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            if params['DOMAIN']=='scalar':
                deconv_layer = nn.Sequential(
                    nn.Linear(params['DIM'], DESCRIBE_DIM*(params['STATE_DEPTH']+1))
                )
            elif params['DOMAIN']=='vector':
                deconv_layer = nn.Sequential(
                    nn.Linear(params['DIM'], DESCRIBE_DIM*(params['STATE_DEPTH']+1)),
                    nn.Sigmoid()
                )

        elif params['DOMAIN']=='image':

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
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
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
        

    def forward(self, noise_v, state_v):

        '''prepare'''
        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
            state_v = state_v.squeeze(1)
        elif params['DOMAIN']=='image':
            # N*D*F*H*W to N*F*D*H*W
            state_v = state_v.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(state_v)
        if params['DOMAIN']=='image':
            temp = x.size()
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.cat_layer(torch.cat([x,noise_v],1))
        x = self.unsqueeze_layer(x)
        if params['DOMAIN']=='image':
            x = x.view(temp)
        x = self.deconv_layer(x)

        '''decompose'''
        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
            stater_v = x.narrow(1,0,DESCRIBE_DIM*params['STATE_DEPTH']).unsqueeze(1)
            prediction_v = x.narrow(1,DESCRIBE_DIM*params['STATE_DEPTH'],DESCRIBE_DIM).unsqueeze(1)
            x = torch.cat([stater_v,prediction_v],1)

        elif params['DOMAIN']=='image':
            # N*F*D*H*W to N*D*F*H*W
            x = x.permute(0,2,1,3,4)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':

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
                D_out_layer(),
            )

        elif params['DOMAIN']=='image':

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
                D_out_layer(),
            )

        if params['GAN_MODE']=='wgan-grad-panish':
            self.conv_layer = conv_layer
            self.squeeze_layer = squeeze_layer
            self.final_layer = final_layer
        else:
            self.conv_layer = torch.nn.DataParallel(conv_layer,GPU)
            self.squeeze_layer = torch.nn.DataParallel(squeeze_layer,GPU)
            self.final_layer = torch.nn.DataParallel(final_layer,GPU)

    def forward(self, state_v, prediction_v):

        '''prepare'''
        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
            state_v = state_v.squeeze(1)
            prediction_v = prediction_v.squeeze(1)
            x = torch.cat([state_v,prediction_v],1)
        elif params['DOMAIN']=='image':
            x = torch.cat([state_v,prediction_v],1)
            # N*D*F*H*W to N*F*D*H*W
            x = x.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(x)
        if params['DOMAIN']=='image':
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.final_layer(x)
        x = x.view(-1)

        return x

class Corrector(nn.Module):

    def __init__(self):
        super(Corrector, self).__init__()

        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':

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

        elif params['DOMAIN']=='image':

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
        if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
            state_v = state_v.squeeze(1)
            prediction_v = prediction_v.squeeze(1)
            x = torch.cat([state_v,prediction_v],1)
        elif params['DOMAIN']=='image':
            x = torch.cat([state_v,prediction_v],1)
            # N*D*F*H*W to N*F*D*H*W
            x = x.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(x)
        if params['DOMAIN']=='image':
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.final_layer(x)
        x = x.view(-1)

        return x

def weights_init_g(m):
    classname = m.__class__.__name__
    sigma = params['G_INIT_SIGMA']
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, sigma)
        if params['DOMAIN']=='vector':
            m.bias.data.normal_(0.0, sigma)
        else:
            m.bias.data.fill_(0.0)
    elif classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, sigma)
    elif classname.find('ConvTranspose3d') != -1:
        m.weight.data.normal_(0.0, sigma)

def weights_init(m):
    classname = m.__class__.__name__
    sigma = 0.02
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, sigma)
        m.bias.data.fill_(0.0)
    elif classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, sigma)
    elif classname.find('ConvTranspose3d') != -1:
        m.weight.data.normal_(0.0, sigma)

def generate_image(iteration):

    '''get data'''
    if params['DOMAIN']=='scalar':
        batch_size = (N_POINTS**2)
    elif params['DOMAIN']=='vector' or params['DOMAIN']=='image':
        batch_size = RESULT_SAMPLE_NUM

    data_fix_state = dataset_iter(
        fix_state=True,
        batch_size=batch_size
    )
    dataset = data_fix_state.next()

    generate_image_with_filter(
        iteration=iteration,
        dataset=dataset,
        gen_basic=True,
        filter_net=None
    )

    if params['FILTER_MODE']=='filter-d-c' or params['FILTER_MODE']=='filter-d':
        generate_image_with_filter(
            iteration=iteration,
            dataset=dataset,
            gen_basic=False,
            filter_net=netD
        )

    if params['FILTER_MODE']=='filter-d-c' or params['FILTER_MODE']=='filter-c':
        generate_image_with_filter(
            iteration=iteration,
            dataset=dataset,
            gen_basic=False,
            filter_net=netC
        )

def transition(cur_x,cur_y,action):

    if params['DATASET']=='1Dflip':
        next_x = np.copy(cur_x)
        next_x[action] = 1.0 - next_x[action]
        return next_x, 0.0

    elif params['DATASET']=='1Dgrid':

        next_x = cur_x
        if action==0:
            next_x = cur_x + 1
        elif action==1:
            next_x = cur_x - 1
        next_x = np.clip(next_x,0,params['GRID_SIZE']-1)
        return next_x, 0.0

    elif params['DATASET']=='2Dgrid':
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

    if params['DATASET']=='1Dflip':
        if params['DOMAIN']=='vector':
            ob = x
        else:
            print(unsupport)
    elif params['DATASET']=='1Dgrid':
        if params['DOMAIN']=='vector':
            ob = np.zeros((params['GRID_SIZE']))
            ob.fill(params['GRID_BACKGROUND'])
            ob[x] = params['GRID_FOREGROUND']
        elif params['DOMAIN']=='image':
            ob = np.zeros((params['IMAGE_SIZE'],params['IMAGE_SIZE']))
            ob.fill(params['GRID_BACKGROUND'])
            ob[x*(params['IMAGE_SIZE']/params['GRID_SIZE']):(x+1)*(params['IMAGE_SIZE']/params['GRID_SIZE']),:] = params['GRID_FOREGROUND']
            ob = np.expand_dims(ob,0)
        else:
            print(unsupport)
    elif params['DATASET']=='2Dgrid':
        if params['DOMAIN']=='scalar':
            ob = [x,y]
        elif params['DOMAIN']=='image':
            ob = np.zeros((params['IMAGE_SIZE'],params['IMAGE_SIZE']))
            ob.fill(params['GRID_BACKGROUND'])
            ob[x*(params['IMAGE_SIZE']/params['GRID_SIZE']):(x+1)*(params['IMAGE_SIZE']/params['GRID_SIZE']),y*(params['IMAGE_SIZE']/params['GRID_SIZE']):(y+1)*(params['IMAGE_SIZE']/params['GRID_SIZE'])] = params['GRID_FOREGROUND']
            ob = np.expand_dims(ob,0)
        else:
            print(unsupport)
    else:
        print(unsupport)

    ob = np.asarray(ob)
    ob = np.expand_dims(ob,0)

    return ob

def get_state(fix_state=False):

    if params['DATASET']=='1Dgrid' or params['DATASET']=='2Dgrid':
        if not fix_state:
            cur_x = np.random.choice(range(params['GRID_SIZE']))
            cur_y = np.random.choice(range(params['GRID_SIZE']))
        else:
            cur_x = FIX_STATE_TO[0]
            cur_y = FIX_STATE_TO[1]

    elif params['DATASET']=='1Dflip':
        if not fix_state:
            cur_x = []
            for i in range(params['GRID_SIZE']):
                cur_x += [np.random.choice([params['GRID_BACKGROUND'],params['GRID_FOREGROUND']])]
        else:
            cur_x = FIX_STATE_TO
        cur_x = np.asarray(cur_x)
        cur_y = 0.0

    return cur_x,cur_y

def get_transition_prob_distribution(images):

    if params['GRID_DETECTION']=='threshold':
        cur_x, cur_y = get_state(fix_state=True)
        accept_num = 0.0
        next_state_dic = []
        for action in range(len(params['GRID_ACTION_DISTRIBUTION'])):
            next_x, next_y = transition(cur_x, cur_y, action)
            ob_next = get_ob(next_x,next_y)
            ob_next = torch.cuda.FloatTensor(ob_next)
            temp = 0.0
            images = images.squeeze()
            ob_next = ob_next.squeeze()
            for b in range(images.size()[0]):
                image = images[b]
                if params['DATASET']=='1Dflip' or params['DATASET']=='1Dgrid':
                    if params['DOMAIN']=='vector':
                        accept = True
                        for ii in range(params['GRID_SIZE']):
                            if np.abs(image[ii]-ob_next[ii]) < params['GRID_ACCEPT']:
                                pass
                            else:
                                accept=False
                                break
                    elif params['DOMAIN']=='image':
                        accept = True
                        for ii in range(params['GRID_SIZE']):
                            if (image[ii*BOX_SIZE:(ii+1)*BOX_SIZE,:]-ob_next[ii*BOX_SIZE:(ii+1)*BOX_SIZE,:]).abs().mean() < params['GRID_ACCEPT']:
                                pass
                            else:
                                accept=False
                                break
                    else:
                        print(unsupport)
                if params['DATASET']=='2Dgrid':
                    if params['DOMAIN']=='image':
                        accept = True
                        for ii_x in range(params['GRID_SIZE']):
                            breaking = False
                            for ii_y in range(params['GRID_SIZE']):
                                if (image[ii_x*BOX_SIZE:(ii_x+1)*BOX_SIZE,ii_y*BOX_SIZE:(ii_y+1)*BOX_SIZE]-ob_next[ii_x*BOX_SIZE:(ii_x+1)*BOX_SIZE,ii_y*BOX_SIZE:(ii_y+1)*BOX_SIZE]).abs().mean() < params['GRID_ACCEPT']:
                                    pass
                                else:
                                    accept=False
                                    breaking=True
                                    break
                            if breaking:
                                break
                    else:
                        print(unsupport)
                if accept:
                    temp += 1.0
                    accept_num += 1.0
            next_state_dic += [temp]
    else:
        print(unsupport)

    next_state_dic = np.asarray(next_state_dic)
    sum_ = np.sum(next_state_dic)
    if not (sum_==0):
        next_state_dic = next_state_dic / sum_

    accept_rate = accept_num / images.size()[0]

    return next_state_dic, accept_rate

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
    l1 = np.squeeze(np.mean(np.abs(dis - np.asarray(params['GRID_ACTION_DISTRIBUTION']))))
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

    state_prediction_gt = torch.Tensor(dataset).cuda()
    state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
    prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

    '''disc_map'''
    if params['DOMAIN']=='scalar':
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(0, params['GRID_SIZE'], N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(0, params['GRID_SIZE'], N_POINTS)[None, :]
        points = points.reshape((-1, 2))
        points = np.expand_dims(points,1)

        disc_map =  netD(
                        state_v = autograd.Variable(state, volatile=True),
                        prediction_v = autograd.Variable(torch.Tensor(points).cuda(), volatile=True)
                    ).cpu().data.numpy()
        x = y = np.linspace(0, params['GRID_SIZE'], N_POINTS)
        disc_map = disc_map.reshape((len(x), len(y))).transpose()
        plt.contour(x, y, disc_map)

        '''narrow to normal batch size'''
        state_prediction_gt = state_prediction_gt.narrow(0,0,RESULT_SAMPLE_NUM)
        state = state.narrow(0,0,RESULT_SAMPLE_NUM)
        prediction_gt = prediction_gt.narrow(0,0,RESULT_SAMPLE_NUM)
        
    prediction_gt_mean = prediction_gt.mean(0,keepdim=True)

    '''prediction_gt'''
    if params['DOMAIN']=='scalar':
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
    if params['DOMAIN']=='scalar':
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
    ).data.narrow(1,1,1)
    
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

    if params['DOMAIN']=='scalar':
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
            if params['DOMAIN']=='vector':
                F_out = F_out.repeat(
                    1,
                    prediction.size()[1],
                    prediction.size()[2])
            elif params['DOMAIN']=='image':
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

    if params['DOMAIN']=='scalar':
        if filter_net is None:
            file_name = ''
        else:
            file_name = 'filtered-by-'+str(filter_net.__class__.__name__)
        plt.savefig(LOGDIR+file_name+'_'+str(iteration)+'.jpg')
        plt_to_vis(
            plt.gcf(),
            win=file_name,
            name=file_name+'_'+str(iteration))

def dataset_iter(fix_state=False, batch_size=params['BATCH_SIZE']):

    while True:
        dataset = []
        for i in xrange(batch_size):

            cur_x, cur_y = get_state(fix_state=fix_state)

            action = np.random.choice(
                range(len(params['GRID_ACTION_DISTRIBUTION'])),
                p=params['GRID_ACTION_DISTRIBUTION']
            )
            
            next_x, next_y = transition(cur_x, cur_y, action)

            ob = get_ob(cur_x,cur_y)
            ob_next = get_ob(next_x,next_y)

            data =  np.concatenate(
                        (ob,ob_next),
                        axis=0
                    )

            dataset.append(data)

        dataset = np.array(dataset, dtype='float32')

        yield dataset

def calc_gradient_penalty(netD, state, interpolates):

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
    gradients = gradients.view(gradients.size()[0],-1)
    gradient_penalty = ((gradients.norm(2, dim=1) - params['GP_TO']) ** 2).mean() * params['LAMBDA']

    return gradient_penalty

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
            if delta<params['GRID_ACCEPT']:
                x_next.push()
                in_cell = True
                break
        
        if not in_cell:
            self.x_next_dic += [TabulatCell(np.copy(x_next_push))]
            # print('Create a cell.')
            self.x_next_dic[-1].push()
        
############################### Definition End ###############################

if params['METHOD']=='tabular':
    tabular_dic = []

elif params['METHOD']=='grl':

    netG = Generator().cuda()
    netD = Discriminator().cuda()
    netC = Corrector().cuda()

    netD.apply(weights_init)
    netC.apply(weights_init)
    netG.apply(weights_init_g)
    print netG
    print netD
    print netC

    if params['OPTIMIZER']=='Adam':
        optimizerD = optim.Adam(netD.parameters(), lr=(1e-4)*params['FASTEN_D'], betas=(0.5, 0.9))
        optimizerC = optim.Adam(netC.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    elif params['OPTIMIZER']=='RMSprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr = (0.00005)*params['FASTEN_D'])
        optimizerC = optim.RMSprop(netC.parameters(), lr = 0.00005)
        optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)

    mse_loss_model = torch.nn.MSELoss(size_average=True)

    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    ones_zeros = torch.cuda.FloatTensor(np.concatenate((np.ones((params['BATCH_SIZE'])),np.zeros((params['BATCH_SIZE']))),0))

    restore_model()

    state_prediction_gt = torch.Tensor(dataset_iter(fix_state=False).next()).cuda()
    state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
    prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)
    alpha_expand = torch.FloatTensor(prediction_gt.size()).cuda()

if params['GAME_MDOE']=='same-start':
    data = dataset_iter(fix_state=True)
elif params['GAME_MDOE']=='full':
    data = dataset_iter(fix_state=False)

logger = lib.plot.logger(LOGDIR,DSP,params_str,CUDA)
iteration = logger.restore()

while True:
    iteration += 1

    if params['METHOD']=='tabular':

        '''get data set'''
        state_prediction_gt = torch.Tensor(data.next()).cuda()
        state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH']).cpu().numpy()
        prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1).cpu().numpy()

        for b in range(np.shape(state)[0]):
            in_tabular = False
            for tabular_i in tabular_dic:
                delta = np.mean(
                    np.abs(state[b] - tabular_i.x),
                    keepdims=False
                )
                if delta<params['GRID_ACCEPT']:
                    tabular_i.push(np.copy(prediction_gt[b]))
                    in_tabular = True
                    # print('Push in a tabular.')
                    break
            if not in_tabular:
                tabular_dic += [Tabular(np.copy(state[b]))]
                # print('Create a tabular.')
                tabular_dic[-1].push(np.copy(prediction_gt[b]))

        x, y = get_state(fix_state=True)
        fix_state_ob = np.expand_dims(get_ob(x,y),0)
        for tabular_i in tabular_dic:
            delta = np.mean(
                np.abs(fix_state_ob - tabular_i.x),
                keepdims=False
            )
            if delta<params['GRID_ACCEPT']:
                tabular_i
                if params['GRID_DETECTION']=='threshold':
                    cur_x, cur_y = get_state(fix_state=True)
                    accept_num = 0.0
                    next_state_dic = []
                    for action in range(len(params['GRID_ACTION_DISTRIBUTION'])):
                        next_x, next_y = transition(cur_x, cur_y, action)
                        ob_next = get_ob(next_x,next_y)
                        in_cell = False
                        temp = 0.0
                        for cell_i in tabular_i.x_next_dic:
                            delta = np.mean(
                                np.abs(ob_next-cell_i.x),
                                keepdims=False
                            )
                            if delta<params['GRID_ACCEPT']:
                                in_cell = True
                                temp = cell_i.count
                                break
                        next_state_dic += [temp]
                else:
                    print(unsupport)
                next_state_dic = np.asarray(next_state_dic)
                sum_ = np.sum(next_state_dic)
                if not (sum_==0):
                    next_state_dic = next_state_dic / sum_

                dis = next_state_dic
                if not (np.sum(dis)==0.0):
                    kl = scipy.stats.entropy(
                        dis,
                        qk=params['GRID_ACTION_DISTRIBUTION'],
                        base=None
                    )
                    logger.plot(
                        'tabular'+'-KL',
                        np.asarray([kl])
                    )
                l1 = np.squeeze(np.mean(np.abs(dis - np.asarray(params['GRID_ACTION_DISTRIBUTION']))))
                logger.plot(
                    'tabular'+'-L1',
                    np.asarray([l1])
                )

                break

        print('[{:<10}]'
            .format(
                iteration
            )
        )


    elif params['METHOD']=='grl':

        ############################
        # (1) Update D network
        ############################

        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in xrange(params['CRITIC_ITERS']):

            if params['GAN_MODE']=='wgan':
                for p in netD.parameters():
                    p.data.clamp_(-0.01, +0.01)

            '''get data set'''
            state_prediction_gt = torch.Tensor(data.next()).cuda()
            state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
            prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)

            '''get generated data'''
            noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
            prediction = netG(
                noise_v = autograd.Variable(noise, volatile=True),
                state_v = autograd.Variable(state, volatile=True)
            ).data.narrow(1,1,1)

            if params['RUINER_MODE']=='use-r':
                noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                prediction_gt = netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(prediction_gt, volatile=True)
                ).data.narrow(1,0,1)
                state_prediction_gt = torch.cat([state,prediction_gt],1)

            netD.zero_grad()

            '''train with real'''
            D_real = netD(
                state_v = autograd.Variable(state),
                prediction_v = autograd.Variable(prediction_gt)
            ).mean()
            D_real.backward(mone)

            '''train with fake'''
            D_fake = netD(
                state_v = autograd.Variable(state),
                prediction_v = autograd.Variable(prediction)
            ).mean()
            D_fake.backward(one)

            GP_cost = [0.0]
            if params['GAN_MODE']=='wgan-grad-panish':
                alpha = torch.rand(params['BATCH_SIZE']).cuda()
                while len(alpha.size())!=len(prediction_gt.size()):
                    alpha = alpha.unsqueeze(1)
                if params['DOMAIN']=='scalar' or params['DOMAIN']=='vector':
                    alpha = alpha.repeat(
                        1,
                        prediction_gt.size()[1],
                        prediction_gt.size()[2]
                    )
                elif params['DOMAIN']=='image':
                    alpha = alpha.repeat(
                        1,
                        prediction_gt.size()[1],
                        prediction_gt.size()[2],
                        prediction_gt.size()[3],
                        prediction_gt.size()[4]
                    )

                interpolates = alpha * prediction_gt + ((1 - alpha) * prediction)
                '''train with gradient penalty'''
                gradient_penalty = calc_gradient_penalty(
                    netD = netD,
                    state = state,
                    interpolates = interpolates
                )
                gradient_penalty.backward()
                GP_cost = gradient_penalty.data.cpu().numpy()

            DC_cost = [0.0]
            if params['GAN_MODE']=='wgan-decade':
                if params['DOMAIN']=='scalar':
                    prediction_uni = torch.cuda.FloatTensor(torch.cat([prediction_gt,prediction],0).size()).uniform_(0.0,params['GRID_SIZE'])
                elif params['DOMAIN']=='image':
                    prediction_uni = torch.cuda.FloatTensor(torch.cat([prediction_gt,prediction],0).size()).uniform_(0.0,1.0)
                D_uni = netD(
                    state_v = autograd.Variable(torch.cat([state,state],0)),
                    prediction_v = autograd.Variable(prediction_uni)
                )
                decade_cost = mse_loss_model(D_uni, autograd.Variable(torch.cuda.FloatTensor(D_uni.size()).fill_(0.0)))
                decade_cost.backward()
                DC_cost = decade_cost.cpu().data.numpy()

            if params['GAN_MODE']=='wgan-grad-panish':
                D_cost = D_fake - D_real + gradient_penalty
            if params['GAN_MODE']=='wgan-decade':
                D_cost = D_fake - D_real + decade_cost
            else:
                D_cost = D_fake - D_real
            D_cost = D_cost.data.cpu().numpy()

            Wasserstein_D = (D_real - D_fake).data.cpu().numpy()

            optimizerD.step()

            C_cost = [0.0]
            if params['FILTER_MODE']=='filter-c' or params['FILTER_MODE']=='filter-d-c':
                netC.zero_grad()

                if params['CORRECTOR_MODE']=='c-normal':

                    C_out_v = netC(
                        state_v = autograd.Variable(torch.cat([state,state],0)),
                        prediction_v = autograd.Variable(torch.cat([prediction_gt,prediction],0))
                    )
                    C_cost_v = torch.nn.functional.binary_cross_entropy(C_out_v,autograd.Variable(ones_zeros))

                elif params['CORRECTOR_MODE']=='c-decade':

                    if params['DOMAIN']=='scalar':
                        prediction_uni = torch.cuda.FloatTensor(prediction_gt.size()).uniform_(0.0,params['GRID_SIZE'])
                    elif params['DOMAIN']=='vector' or params['DOMAIN']=='image':
                        prediction_uni = torch.cuda.FloatTensor(prediction_gt.size()).uniform_(0.0,1.0)
                    C_out_v = netC(
                        state_v = autograd.Variable(torch.cat([state,state],0)),
                        prediction_v = autograd.Variable(torch.cat([prediction_gt,prediction_uni],0))
                    )
                    C_cost_v = mse_loss_model(C_out_v,autograd.Variable(ones_zeros))

                C_cost_v.backward()
                C_cost = C_cost_v.cpu().data.numpy()
                optimizerC.step()

        if params['GAN_MODE']=='wgan-gravity':
            for p in netD.parameters():
                p.data = p.data * (1.0-0.0001)

        if params['CORRECTOR_MODE']=='c-decade':
            for p in netC.parameters():
                p.data = p.data * (1.0-0.01)

        if params['GAN_MODE']=='wgan-grad-panish':
            logger.plot('GP_cost', GP_cost)
        if params['GAN_MODE']=='wgan-decade':
            logger.plot('DC_cost', DC_cost)
        if params['FILTER_MODE']=='filter-c' or params['FILTER_MODE']=='filter-d-c':
            logger.plot('C_cost', C_cost)
        logger.plot('D_cost', D_cost)
        logger.plot('W_dis', Wasserstein_D)

        ############################
        # (2) Control R
        ############################

        if params['RUINER_MODE']=='use-r':
            if Wasserstein_D[0] > params['TARGET_W_DISTANCE']:
                update_type = 'g'
            else:
                update_type = 'r'
        elif params['RUINER_MODE']=='none-r':
            update_type = 'g'
        elif params['RUINER_MODE']=='test-r':
            update_type = 'r'

        ############################
        # (3) Update G network or R
        ###########################

        state_prediction_gt = torch.Tensor(data.next()).cuda()
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
            ).narrow(1,1,1)

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

            if params['GAME_MDOE']=='full':

                noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                stater_v = netG(
                    noise_v = autograd.Variable(noise),
                    state_v = autograd.Variable(state)
                ).narrow(1,0,1)

                R = mse_loss_model(stater_v, autograd.Variable(state.narrow(1,params['STATE_DEPTH']-1,1)))

            elif params['GAME_MDOE']=='same-start':

                noise = torch.randn(params['BATCH_SIZE'], params['NOISE_SIZE']).cuda()
                prediction_gt_r_v = netG(
                    noise_v = autograd.Variable(noise),
                    state_v = autograd.Variable(prediction_gt)
                ).narrow(1,0,1)

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

        if iteration % LOG_INTER == 5:
            torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
            torch.save(netC.state_dict(), '{0}/netC.pth'.format(LOGDIR))
            torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))
            generate_image(iteration)
        
        print('[{:<10}] W_cost:{:2.4f} GP_cost:{:2.4f} D_cost:{:2.4f} G_R:{} G_cost:{:2.4f} R_cost:{:2.4f} C_cost:{:2.4f}'
            .format(
                iteration,
                Wasserstein_D[0],
                GP_cost[0],
                D_cost[0],
                G_R,
                G_cost[0],
                R_cost[0],
                C_cost[0]
            )
        )

    if iteration % LOG_INTER == 5:
        logger.flush()

    logger.tick()

    
