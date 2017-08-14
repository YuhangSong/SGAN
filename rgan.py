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

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
from PIL import Image

import torchvision.utils as vutils
import visdom
vis = visdom.Visdom()

torch.manual_seed(4213)

GPU = range(torch.cuda.device_count())
print('Using GPU:'+str(GPU))

params = {}

def add_parameters(**kwargs):
    params.update(kwargs)

add_parameters(EXP = 'd_filter_18')
add_parameters(DATASET = '2grid') # 2grid
add_parameters(GAME_MDOE = 'full') # same-start, full
add_parameters(DOMAIN = 'image') # scalar, image
add_parameters(GAN_MODE = 'wgan-grad-panish') # wgan, wgan-grad-panish, wgan-gravity, wgan-decade
add_parameters(RUINER_MODE = 'use-r') # none-r, use-r, test-r
add_parameters(FILTER_MODE = 'filter-d-c') # none-f, filter-c, filter-d, filter-d-c
add_parameters(CORRECTOR_MODE = 'c-decade') # c-normal, c-decade
add_parameters(OPTIMIZER = 'Adam') # Adam, RMSprop
add_parameters(INIT_SIGMA = 0.00002)
add_parameters(FASTEN_D = 10)

if params['DATASET']=='2grid':
    add_parameters(GRID_SIZE = 5)
    add_parameters(GRID_BACKGROUND = 0.1)
    add_parameters(GRID_FOREGROUND = 0.9)
    add_parameters(GRID_ACTION_DISTRIBUTION = [0.25,0.25,0.25,0.25])
    FIX_STATE_TO = [params['GRID_SIZE']/2,params['GRID_SIZE']/2]

if params['DOMAIN']=='scalar':
    add_parameters(DIM = 512)
    add_parameters(NOISE_SIZE = 2)
    add_parameters(LAMBDA = 0.1) # Smaller lambda seems to help for toy tasks specifically
    add_parameters(BATCH_SIZE = 256)
    add_parameters(TARGET_W_DISTANCE = 0.2)
    add_parameters(STATE_DEPTH = 1)

    LOG_INTER = 100

elif params['DOMAIN']=='image':
    add_parameters(DIM = 128)
    add_parameters(NOISE_SIZE = 128)
    add_parameters(LAMBDA = 10)
    add_parameters(BATCH_SIZE = 64)
    add_parameters(TARGET_W_DISTANCE = 0.1)
    add_parameters(STATE_DEPTH = 1)
    
    add_parameters(FEATURE = 1)
    add_parameters(IMAGE_SIZE = 32)

    LOG_INTER = 100
    
    GRID_BOX_SIZE = params['IMAGE_SIZE'] / params['GRID_SIZE']


add_parameters(CRITIC_ITERS = 5)  # How many critic iterations per generator iteration
N_POINTS = 128

RESULT_SAMPLE_NUM = 2000
FILTER_RATE = 0.5

print('####################################################################')
DSP = ''
for key,val in params.items():
    print(key+' >> '+str(val))
    DSP += key+'_'+str(val).replace('.','_').replace(',','_').replace(' ','_')+'/'
print('####################################################################')

BASIC = '../../result/'
LOGDIR = BASIC+DSP

subprocess.call(["mkdir", "-p", LOGDIR])

############################### Definition Start ###############################

def log_img(x,name,iteration):
    x = x.squeeze(1)
    vutils.save_image(x, LOGDIR+name+'_'+str(iteration)+'.png')
    vis.images( x.cpu().numpy(),
                win=DSP+name,
                opts=dict(caption=DSP+name+'_'+str(iteration)))

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
                win=win,
                opts=dict(title=name))

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

        if params['DOMAIN']=='scalar':

            conv_layer = nn.Sequential(
                nn.Linear(2, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.conv_layer = conv_layer

            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
            )
            self.cat_layer = cat_layer

            deconv_layer = nn.Sequential(
                nn.Linear(params['DIM'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(params['DIM'], 2*(params['STATE_DEPTH']+1)),
            )
            self.deconv_layer = deconv_layer

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
            self.conv_layer = nn.DataParallel(conv_layer,GPU)

            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*4*4, params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.squeeze_layer = nn.DataParallel(squeeze_layer,GPU)

            cat_layer = nn.Sequential(
                nn.Linear(params['DIM']+params['NOISE_SIZE'], params['DIM']),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.cat_layer = nn.DataParallel(cat_layer,GPU)

            unsqueeze_layer = nn.Sequential(
                nn.Linear(params['DIM'], 256*1*4*4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.unsqueeze_layer = nn.DataParallel(unsqueeze_layer,GPU)

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
            self.deconv_layer = torch.nn.DataParallel(deconv_layer,GPU)

    def forward(self, noise_v, state_v):

        '''prepare'''
        if params['DOMAIN']=='scalar':
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
        if params['DOMAIN']=='image':
            x = self.unsqueeze_layer(x)
            x = x.view(temp)
        x = self.deconv_layer(x)

        '''decompose'''
        if params['DOMAIN']=='scalar':
            stater_v = x.narrow(1,0,2).unsqueeze(1)
            prediction_v = x.narrow(1,2,2).unsqueeze(1)
            x = torch.cat([stater_v,prediction_v],1)
        else:
            # N*F*D*H*W to N*D*F*H*W
            x = x.permute(0,2,1,3,4)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        if params['DOMAIN']=='scalar':

            conv_layer = nn.Sequential(
                nn.Linear(2+2, params['DIM']),
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
                # nn.InstanceNorm3d(64),
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
                # nn.InstanceNorm3d(128),
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
                # nn.InstanceNorm3d(256),
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
        if params['DOMAIN']=='scalar':
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

        if params['DOMAIN']=='scalar':

            conv_layer = nn.Sequential(
                nn.Linear(2+2, params['DIM']),
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

            final_layer = nn.Sequential(
                nn.Linear(params['DIM'], 1),
                nn.Sigmoid(),
            )

        self.conv_layer = torch.nn.DataParallel(conv_layer,GPU)
        self.squeeze_layer = torch.nn.DataParallel(squeeze_layer,GPU)
        self.final_layer = torch.nn.DataParallel(final_layer,GPU)

    def forward(self, state_v, prediction_v):

        '''prepare'''
        if params['DOMAIN']=='scalar':
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
    print('----------------------------------')
    print('class name:'+str(classname))
    sigma = params['INIT_SIGMA']
    if classname.find('Linear') != -1:
        print('>>> find Linear')
        m.weight.data.normal_(0.0, sigma)
        m.bias.data.fill_(0.0)
    elif classname.find('Conv3d') != -1:
        print('>>> find Conv3d')
        m.weight.data.normal_(0.0, sigma)
    elif classname.find('ConvTranspose3d') != -1:
        print('>>> find ConvTranspose3d')
        m.weight.data.normal_(0.0, sigma)
    print('----------------------------------')

def weights_init(m):
    classname = m.__class__.__name__
    print('----------------------------------')
    print('class name:'+str(classname))
    sigma = 0.02
    if classname.find('Linear') != -1:
        print('>>> find Linear')
        m.weight.data.normal_(0.0, sigma)
        m.bias.data.fill_(0.0)
    elif classname.find('Conv3d') != -1:
        print('>>> find Conv3d')
        m.weight.data.normal_(0.0, sigma)
    elif classname.find('ConvTranspose3d') != -1:
        print('>>> find ConvTranspose3d')
        m.weight.data.normal_(0.0, sigma)
    print('----------------------------------')

def generate_image(iteration):

    '''get data'''
    if params['DOMAIN']=='scalar':
        batch_size = (N_POINTS**2)
    elif params['DOMAIN']=='image':
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

def get_transition_prob_distribution(image):
    image = image.squeeze()
    cur_x = FIX_STATE_TO[0]
    cur_y = FIX_STATE_TO[1]
    next_state_dic = []
    for action in range(len(params['GRID_ACTION_DISTRIBUTION'])):
        x, y = grid_transition(cur_x, cur_y, action)
        temp = image[x*(params['IMAGE_SIZE']/params['GRID_SIZE']):(x+1)*(params['IMAGE_SIZE']/params['GRID_SIZE']),y*(params['IMAGE_SIZE']/params['GRID_SIZE']):(y+1)*(params['IMAGE_SIZE']/params['GRID_SIZE'])].sum()
        next_state_dic += [temp]
    next_state_dic = np.asarray(next_state_dic)
    next_state_dic = next_state_dic / np.sum(next_state_dic)
    return next_state_dic

def plot_kl(image,name):
    dis = get_transition_prob_distribution(image)
    kl = scipy.stats.entropy(
        dis,
        qk=params['GRID_ACTION_DISTRIBUTION'],
        base=None
    )
    logger.plot(
        name,
        kl
    )

def generate_image_with_filter(iteration,dataset,gen_basic=False,filter_net=None):

    plt.clf()

    state_prediction_gt = torch.Tensor(dataset).cuda()
    state = state_prediction_gt.narrow(1,0,1)
    prediction_gt = state_prediction_gt.narrow(1,1,1)

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
        
    prediction_gt_mean = prediction_gt.mean(0)

    '''prediction_gt'''
    if params['DOMAIN']=='scalar':
        plt.scatter(
            prediction_gt.squeeze(1).cpu().numpy()[:, 0], 
            prediction_gt.squeeze(1).cpu().numpy()[:, 1],
            c='orange', 
            marker='+', 
            alpha=0.5
        )
    elif params['DOMAIN']=='image':
        if gen_basic:
            log_img(prediction_gt,'prediction_gt',iteration)
            prediction_gt_mean = prediction_gt.mean(0)
            log_img(prediction_gt_mean,'prediction_gt_mean',iteration)
            plot_kl(image=prediction_gt_mean,
                name='prediction_gt_mean_kl'
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
    elif params['DOMAIN']=='image':
        if gen_basic:
            noise = torch.randn((RESULT_SAMPLE_NUM), params['NOISE_SIZE']).cuda()
            prediction_gt_r = netG(
                noise_v = autograd.Variable(noise, volatile=True),
                state_v = autograd.Variable(prediction_gt, volatile=True)
            ).data.narrow(1,0,1)
            log_img(prediction_gt_r,'prediction_gt_r',iteration)
            prediction_gt_r_mean = prediction_gt_r.mean(0)
            log_img(prediction_gt_r_mean,'prediction_gt_r_mean',iteration)
            plot_kl(image=prediction_gt_r_mean,
                name='prediction_gt_r_mean_kl'
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

    elif params['DOMAIN']=='image':

        if filter_net is None:

            log_img(prediction,'prediction',iteration)
            prediction_mean = prediction.mean(0)
            log_img(
                prediction_mean,
                'prediction-mean',
                iteration
            )

        else:

            F_out_numpy = F_out.cpu().numpy()

            while len(F_out.size())!=len(prediction.size()):
                F_out = F_out.unsqueeze(1)
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
            filtered_prediction_mean = filtered_prediction.mean(0)
            log_img(
                x=filtered_prediction_mean,
                name='prediction-mean-filtered-by-'+str(filter_net.__class__.__name__),
                iteration=iteration
            )

            plot_kl(image=filtered_prediction_mean,
                name='prediction-kl-filtered-by-'+str(filter_net.__class__.__name__)
            )

    if params['DOMAIN']=='scalar':
        if filter_net is None:
            file_name = ''
        else:
            file_name = 'filtered-by-'+str(filter_net.__class__.__name__)
        plt.savefig(LOGDIR+file_name+'_'+str(iteration)+'.jpg')
        plt_to_vis(plt.gcf(),DSP+file_name,DSP+file_name+'_'+str(iteration))

def grid_transition(cur_x,cur_y,action):
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
    next_x = np.clip(next_x,0,params['GRID_SIZE'])
    next_y = np.clip(next_y,0,params['GRID_SIZE'])
    return next_x,next_y

def dataset_iter(fix_state=False, batch_size=params['BATCH_SIZE']):

    if params['DATASET']=='2grid':

        scale = 2.
        centers = [
            (1, 1),
            (-1, -1)
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in xrange(batch_size):
                if not fix_state:
                    cur_x = np.random.choice(range(params['GRID_SIZE']))
                    cur_y = np.random.choice(range(params['GRID_SIZE']))
                else:
                    cur_x = FIX_STATE_TO[0]
                    cur_y = FIX_STATE_TO[1]
                action = np.random.choice(
                    range(len(params['GRID_ACTION_DISTRIBUTION'])),
                    p=params['GRID_ACTION_DISTRIBUTION']
                )
                
                next_x, next_y = grid_transition(cur_x,cur_y,action)

                if params['DOMAIN']=='scalar':
                    data = np.array([[cur_x,cur_y],
                                     [next_x,next_y]])
                elif params['DOMAIN']=='image':
                    def to_ob(x,y):
                        ob = np.zeros((params['IMAGE_SIZE'],params['IMAGE_SIZE']))
                        ob.fill(params['GRID_BACKGROUND'])
                        ob[x*(params['IMAGE_SIZE']/params['GRID_SIZE']):(x+1)*(params['IMAGE_SIZE']/params['GRID_SIZE']),y*(params['IMAGE_SIZE']/params['GRID_SIZE']):(y+1)*(params['IMAGE_SIZE']/params['GRID_SIZE'])] = params['GRID_FOREGROUND']
                        ob = np.expand_dims(ob,0)
                        ob = np.expand_dims(ob,0)
                        return ob
                    data =  np.concatenate(
                                (to_ob(cur_x,cur_y),to_ob(next_x,next_y)),
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * params['LAMBDA']

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

############################### Definition End ###############################

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
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerC = optim.Adam(netC.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
elif params['OPTIMIZER']=='RMSprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr = 0.00005)
    optimizerC = optim.RMSprop(netC.parameters(), lr = 0.00005)
    optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)

mse_loss_model = torch.nn.MSELoss(size_average=True)

one = torch.FloatTensor([1]).cuda()
mone = one * -1
ones_zeros = torch.cuda.FloatTensor(np.concatenate((np.ones((params['BATCH_SIZE'])),np.zeros((params['BATCH_SIZE']))),0))

if params['GAME_MDOE']=='same-start':
    data = dataset_iter(fix_state=True)
elif params['GAME_MDOE']=='full':
    data = dataset_iter(fix_state=False)

restore_model()
logger = lib.plot.logger(LOGDIR,DSP)

state_prediction_gt = torch.Tensor(data.next()).cuda()
state = state_prediction_gt.narrow(1,0,params['STATE_DEPTH'])
prediction_gt = state_prediction_gt.narrow(1,params['STATE_DEPTH'],1)
alpha_expand = torch.FloatTensor(prediction_gt.size()).cuda()

iteration = -1
while True:
    iteration += 1

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
            if params['DOMAIN']=='scalar':
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
                elif params['DOMAIN']=='image':
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

    if iteration % LOG_INTER == 2:
        torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
        torch.save(netC.state_dict(), '{0}/netC.pth'.format(LOGDIR))
        torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))
        generate_image(iteration)
        logger.flush()
    
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

    logger.tick()
