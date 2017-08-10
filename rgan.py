import os, sys

sys.path.append(os.getcwd())

import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

GPU = range(2)

EXP = 'baseline_86'

DATASET = '2grid' # 2grid
GAME_MDOE = 'full' # same-start, full
DOMAIN = 'image' # scalar, image
GAN_MODE = 'wgan-decade' # wgan, wgan-grad-panish, wgan-gravity, wgan-decade
R_MODE = 'use-r' # use-r, none-r, test-r

DSP = EXP+'/'+DATASET+'/'+GAME_MDOE+'/'+DOMAIN+'/'+GAN_MODE+'/'+R_MODE+'/'
BASIC = '../../result/'
LOGDIR = BASIC+DSP

STATE_DEPTH = 1
if DOMAIN=='scalar':
    DIM = 512
    NOISE_SIZE = 2
    LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
    BATCH_SIZE = 256
elif DOMAIN=='image':
    DIM = 64
    IMAGE_SIZE = 32
    FEATURE = 1
    NOISE_SIZE = 128
    LAMBDA = 10
    BATCH_SIZE = 50

if DATASET=='2grid':
    GRID_SIZE = 5

CRITIC_ITERS = 5  # How many critic iterations per generator iteration
ITERS = 1000000000  # how many generator iterations to train for
use_cuda = True
N_POINTS = 128
GRID_BACKGROUND = 0.2
GRID_FOREGROUND = 0.8

def prepare_dir():
    subprocess.call(["mkdir", "-p", LOGDIR])
prepare_dir()

# ==================Definition Start======================

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
        if GAN_MODE=='wgan-gravity':
            for i in range(x.size()[0]):
                if x[i].data.cpu().numpy()[0]>0.0:
                    x[i] = torch.log(x[i]+1.0)
                elif x[i].data.cpu().numpy()[0]<0.0:
                    x[i] = -torch.log(-x[i]+1.0)
        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        if DOMAIN=='scalar':

            conv_layer = nn.Sequential(
                nn.Linear(2, DIM),
                nn.ReLU(True),
                nn.Linear(DIM, DIM),
                nn.ReLU(True),
            )
            self.conv_layer = conv_layer

            cat_layer = nn.Sequential(
                nn.Linear(DIM+NOISE_SIZE, DIM),
            )
            self.cat_layer = cat_layer

            deconv_layer = nn.Sequential(
                nn.Linear(DIM, DIM),
                nn.ReLU(True),
                nn.Linear(DIM, 2*(STATE_DEPTH+1)),
            )
            self.deconv_layer = deconv_layer

        elif DOMAIN=='image':

            conv_layer = nn.Sequential(

                # FEATURE*1*32*32

                nn.Conv3d(
                    in_channels=FEATURE,
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.BatchNorm3d(64),
                nn.ReLU(True),

                # 64*1*16*16

                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.BatchNorm3d(128),
                nn.ReLU(True),

                # 128*1*8*8

                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.BatchNorm3d(256),
                nn.ReLU(True),

                # 256*1*4*4
            )
            self.conv_layer = nn.DataParallel(conv_layer,GPU)

            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*4*4, DIM),
                nn.ReLU(True),
            )
            self.squeeze_layer = nn.DataParallel(squeeze_layer,GPU)

            cat_layer = nn.Sequential(
                nn.Linear(DIM+NOISE_SIZE, DIM),
                nn.ReLU(True),
            )
            self.cat_layer = nn.DataParallel(cat_layer,GPU)

            unsqueeze_layer = nn.Sequential(
                nn.Linear(DIM, 256*1*4*4),
                nn.ReLU(True),
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
                    bias=True
                ),
                nn.BatchNorm3d(128),
                nn.ReLU(True),

                # 128*2*8*8

                nn.ConvTranspose3d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.BatchNorm3d(64),
                nn.ReLU(True),

                # 64*2*16*16

                nn.ConvTranspose3d(
                    in_channels=64,
                    out_channels=FEATURE,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                # nn.Tanh(),
                nn.Sigmoid()

                # FEATURE*2*32*32

                
            )
            self.deconv_layer = torch.nn.DataParallel(deconv_layer,GPU)

    def forward(self, noise_v, state_v):

        '''prepare'''
        if DOMAIN=='scalar':
            state_v = state_v.squeeze(1)
        elif DOMAIN=='image':
            # N*D*F*H*W to N*F*D*H*W
            state_v = state_v.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(state_v)
        if DOMAIN=='image':
            temp = x.size()
            x = x.view(x.size()[0], -1)
            x = self.squeeze_layer(x)
        x = self.cat_layer(torch.cat([x,noise_v],1))
        if DOMAIN=='image':
            x = self.unsqueeze_layer(x)
            x = x.view(temp)
        x = self.deconv_layer(x)

        '''decompose'''
        if DOMAIN=='scalar':
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

        if DOMAIN=='scalar':

            conv_layer = nn.Sequential(
                nn.Linear(2+2, DIM),
                nn.ReLU(True),
                nn.Linear(DIM, DIM),
                nn.ReLU(True),
            )

            squeeze_layer = nn.Sequential(
                nn.Linear(DIM, DIM),
                nn.ReLU(True),
            )

            final_layer = nn.Sequential(
                nn.Linear(DIM, 1),
                D_out_layer(),
            )


        elif DOMAIN=='image':

            conv_layer =    nn.Sequential(

                # FEATURE*2*32*32

                nn.Conv3d(
                    in_channels=FEATURE,
                    out_channels=64,
                    kernel_size=(2,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.InstanceNorm3d(64),
                nn.ReLU(True),

                # 64*1*16*16

                nn.Conv3d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.InstanceNorm3d(128),
                nn.ReLU(True),

                # 128*1*8*8

                nn.Conv3d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(1,4,4),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=True
                ),
                nn.InstanceNorm3d(256),
                nn.ReLU(True),

                # 256*1*4*4
            )

            squeeze_layer = nn.Sequential(
                nn.Linear(256*1*4*4, DIM),
                nn.ReLU(True),
            )

            final_layer = nn.Sequential(
                nn.Linear(DIM, 1),
                D_out_layer(),
            )

        if GAN_MODE=='wgan-grad-panish':
            self.conv_layer = conv_layer
            self.squeeze_layer = squeeze_layer
            self.final_layer = final_layer
        else:
            self.conv_layer = torch.nn.DataParallel(conv_layer,GPU)
            self.squeeze_layer = torch.nn.DataParallel(squeeze_layer,GPU)
            self.final_layer = torch.nn.DataParallel(final_layer,GPU)

    def forward(self, state_v, prediction_v):

        '''prepare'''
        if DOMAIN=='scalar':
            state_v = state_v.squeeze(1)
            prediction_v = prediction_v.squeeze(1)
            x = torch.cat([state_v,prediction_v],1)
        elif DOMAIN=='image':
            x = torch.cat([state_v,prediction_v],1)
            # N*D*F*H*W to N*F*D*H*W
            x = x.permute(0,2,1,3,4)

        '''forward'''
        x = self.conv_layer(x)
        if DOMAIN=='image':
            x = x.view(x.size()[0], -1)
        x = self.squeeze_layer(x)
        x = self.final_layer(x)
        x = x.view(-1)

        return x


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    print('----------------------------------')
    print('class name:'+str(classname))
    sigma = 0.02
    if classname.find('Linear') != -1:
        print('\tfind Linear')
        m.weight.data.normal_(0.0, sigma)
        m.bias.data.fill_(0.0)
    elif classname.find('Conv3d') != -1:
        print('\tfind Conv3d')
        m.weight.data.normal_(0.0, sigma)
    elif classname.find('ConvTranspose3d') != -1:
        print('\tfind ConvTranspose3d')
        m.weight.data.normal_(0.0, sigma)
    print('----------------------------------')

frame_index = [0]

def generate_image():
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    plt.clf()

    '''get data'''
    if DOMAIN=='scalar':
        bs = (N_POINTS**2)
    elif DOMAIN=='image':
        bs = BATCH_SIZE
    data_fix_state = inf_train_gen(fix_state=True,batch_size=bs)
    state_prediction_gt = torch.Tensor(data_fix_state.next()).cuda()
    state = state_prediction_gt.narrow(1,0,1)
    prediction_gt = state_prediction_gt.narrow(1,1,1)

    '''disc_map'''
    if DOMAIN=='scalar':
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(0, GRID_SIZE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(0, GRID_SIZE, N_POINTS)[None, :]
        points = points.reshape((-1, 2))
        points = np.expand_dims(points,1)

        disc_map =  netD(
                        state_v = autograd.Variable(state, volatile=True),
                        prediction_v = autograd.Variable(torch.Tensor(points).cuda(), volatile=True)
                    ).cpu().data.numpy()
        x = y = np.linspace(0, GRID_SIZE, N_POINTS)
        disc_map = disc_map.reshape((len(x), len(y))).transpose()
        plt.contour(x, y, disc_map)

        '''narrow to normal batch size'''
        state_prediction_gt = state_prediction_gt.narrow(0,0,BATCH_SIZE)
        state = state.narrow(0,0,BATCH_SIZE)
        prediction_gt = prediction_gt.narrow(0,0,BATCH_SIZE)

    frame = str(frame_index[0])
    def log_img(x,name,frame):
        x=x.squeeze(1)
        vutils.save_image(x, LOGDIR+name+'_'+frame+'.png')
        vis.images( x.cpu().numpy(),
                    win=DSP+name,
                    opts=dict(caption=DSP+name+'_'+frame))

    '''r'''
    samples = prediction_gt
    if DOMAIN=='scalar':
        samples = prediction_gt.squeeze(1).cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], c='orange', marker='+', alpha=0.5)
    elif DOMAIN=='image':
        log_img(samples,'r',frame)

    '''e'''
    noise = torch.randn((BATCH_SIZE), NOISE_SIZE).cuda()
    samples =   netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(prediction_gt, volatile=True)
                ).data.narrow(1,STATE_DEPTH-1,1)
    if DOMAIN=='scalar':
        samples = samples.squeeze(1).cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], c='blue', marker='+', alpha=0.5)
    elif DOMAIN=='image':
        log_img(samples,'e',frame)

    '''g'''
    noise = torch.randn((BATCH_SIZE), NOISE_SIZE).cuda()
    samples =   netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(state, volatile=True)
                ).data.narrow(1,STATE_DEPTH,1)
    if DOMAIN=='scalar':
        samples = samples.squeeze(1).cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+', alpha=0.5)
    elif DOMAIN=='image':
        log_img(samples,'g',frame)

    if DOMAIN=='scalar':
        plt.savefig(LOGDIR+'/'+'frame'+str(frame_index[0])+'.jpg')
        plt_to_vis(plt.gcf(),DSP+'vis',DSP+'fram_'+str(frame_index[0]))

    frame_index[0] += 1


# Dataset iterator
def inf_train_gen(fix_state=False, fix_state_to=[GRID_SIZE/2,GRID_SIZE/2], batch_size=BATCH_SIZE):

    if DATASET=='2grid':

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
                    cur_x = random.choice(range(GRID_SIZE))
                    cur_y = random.choice(range(GRID_SIZE))
                else:
                    cur_x = fix_state_to[0]
                    cur_y = fix_state_to[1]
                action = random.choice(range(4))
                next_x = cur_x
                next_y = cur_y
                if action==0:
                    next_x = cur_x + 1
                elif action==2:
                    next_x = cur_x - 1
                elif action==1:
                    next_y = cur_y + 1
                elif action==3:
                    next_y = cur_y - 1
                next_x = np.clip(next_x,0,GRID_SIZE)
                next_y = np.clip(next_y,0,GRID_SIZE)

                if DOMAIN=='scalar':
                    data = np.array([[cur_x,cur_y],
                                     [next_x,next_y]])
                elif DOMAIN=='image':
                    def to_ob(x,y):
                        ob = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
                        ob.fill(GRID_BACKGROUND)
                        ob[x*(IMAGE_SIZE/GRID_SIZE):(x+1)*(IMAGE_SIZE/GRID_SIZE),y*(IMAGE_SIZE/GRID_SIZE):(y+1)*(IMAGE_SIZE/GRID_SIZE)] = GRID_FOREGROUND
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


def calc_gradient_penalty(netD, state, prediction_gt, prediction):

    alpha = torch.rand(BATCH_SIZE).cuda()
    alpha_expand = torch.FloatTensor(prediction_gt.size()).cuda()
    for i in range(alpha.size()[0]):
        alpha_expand[i].fill_(alpha[i])

    interpolates = alpha_expand * prediction_gt + ((1 - alpha_expand) * prediction)

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

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print netG
print netD

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

mse_loss_model = torch.nn.MSELoss(size_average=True)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

if GAME_MDOE=='same-start':
    data = inf_train_gen(fix_state=True)
elif GAME_MDOE=='full':
    data = inf_train_gen(fix_state=False)

print('Trying load models....')
try:
    netD.load_state_dict(torch.load('{0}/netD.pth'.format(LOGDIR)))
    print('Previous checkpoint for netD founded')
except Exception, e:
    print('Previous checkpoint for netD unfounded')
try:
    netG.load_state_dict(torch.load('{0}/netG.pth'.format(LOGDIR)))
    print('Previous checkpoint for netC founded')
except Exception, e:
    print('Previous checkpoint for netC unfounded')

# lib.plot.restore(LOGDIR,DSP)

iteration = -1
while True:
    iteration += 1

    ############################
    # (1) Update D network
    ###########################

    for p in netD.parameters():
        p.requires_grad = True

    for iter_d in xrange(CRITIC_ITERS):

        if GAN_MODE=='wgan':
            for p in netD.parameters():
                p.data.clamp_(-0.01, +0.01)

        '''get data set'''
        state_prediction_gt = torch.Tensor(data.next()).cuda()
        state = state_prediction_gt.narrow(1,0,STATE_DEPTH)
        prediction_gt = state_prediction_gt.narrow(1,STATE_DEPTH,1)

        '''get generated data'''
        noise = torch.randn(BATCH_SIZE, NOISE_SIZE).cuda()
        stater_prediction = netG(
            noise_v = autograd.Variable(noise, volatile=True),
            state_v = autograd.Variable(state, volatile=True)
        ).data
        prediction = stater_prediction.narrow(1,STATE_DEPTH,1)

        if R_MODE=='use-r':
            noise = torch.randn(BATCH_SIZE, NOISE_SIZE).cuda()
            prediction_gt = netG(
                noise_v = autograd.Variable(noise, volatile=True),
                state_v = autograd.Variable(prediction_gt, volatile=True)
            ).data.narrow(1,STATE_DEPTH-1,1)
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

        if GAN_MODE=='wgan-grad-panish':
            '''train with gradient penalty'''
            gradient_penalty = calc_gradient_penalty(
                netD = netD,
                state = state,
                prediction_gt = prediction_gt,
                prediction = prediction
            )
            gradient_penalty.backward()

        if GAN_MODE=='wgan-decade':
            if DOMAIN=='scalar':
                prediction_uni = torch.cuda.FloatTensor(torch.cat([prediction_gt,prediction],0).size()).uniform_(0.0,GRID_SIZE)
            elif DOMAIN=='image':
                prediction_uni = torch.cuda.FloatTensor(torch.cat([prediction_gt,prediction],0).size()).uniform_(GRID_BACKGROUND,GRID_FOREGROUND)
            D_uni = netD(
                state_v = autograd.Variable(torch.cat([state,state],0)),
                prediction_v = autograd.Variable(prediction_uni)
            )
            decade_cost = mse_loss_model(D_uni, autograd.Variable(torch.cuda.FloatTensor(D_uni.size()).fill_(0.0)))
            decade_cost.backward()

        if GAN_MODE=='wgan-grad-panish':
            D_cost = D_fake - D_real + gradient_penalty
        if GAN_MODE=='wgan-decade':
            D_cost = D_fake - D_real + decade_cost
        else:
            D_cost = D_fake - D_real

        Wasserstein_D = D_real - D_fake

        optimizerD.step()

    if GAN_MODE=='wgan-gravity':
        for p in netD.parameters():
            p.data = p.data * (1.0-0.0001)

    if GAN_MODE=='wgan-grad-panish':
        lib.plot.plot('GP_cost', gradient_penalty.cpu().data.numpy())
    if GAN_MODE=='wgan-decade':
        lib.plot.plot('decade_cost', decade_cost.cpu().data.numpy())
    lib.plot.plot('D_cost', D_cost.cpu().data.numpy())
    lib.plot.plot('W_dis', Wasserstein_D.cpu().data.numpy())

    ############################
    # (2) Control R
    ############################

    if R_MODE=='use-r':
        if Wasserstein_D.cpu().data.numpy()[0] > 0.0:
            update_type = 'g'
        else:
            update_type = 'r'
    elif R_MODE=='none-r':
        update_type = 'g'
    elif R_MODE=='test-r':
        update_type = 'r'

    ############################
    # (3) Update G network or R
    ###########################

    state_prediction_gt = torch.Tensor(data.next()).cuda()
    state = state_prediction_gt.narrow(1,0,STATE_DEPTH)
    prediction_gt = state_prediction_gt.narrow(1,STATE_DEPTH,1)

    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, NOISE_SIZE).cuda()
    stater_prediction_v = netG(
                            noise_v = autograd.Variable(noise),
                            state_v = autograd.Variable(state)
                        )
    stater_v = stater_prediction_v.narrow(1,0,STATE_DEPTH)
    prediction_v = stater_prediction_v.narrow(1,STATE_DEPTH,1)

    if update_type=='g':
        '''to avoid computation'''
        for p in netD.parameters():
            p.requires_grad = False
        G = netD(
                state_v = autograd.Variable(state),
                prediction_v = prediction_v
            ).mean()
        G_cost = -G
        lib.plot.plot('G_cost', G_cost.cpu().data.numpy())
        lib.plot.plot('G_R', np.asarray([1.0]))
        G.backward(mone)
    elif update_type=='r':
        R = mse_loss_model(stater_v, autograd.Variable(state))
        R_cost = R
        lib.plot.plot('R_cost', R_cost.cpu().data.numpy())
        lib.plot.plot('G_R', np.asarray([-1.0]))
        R.backward()
    
    optimizerG.step()

    ############################
    # (4) Log summary
    ############################

    lib.plot.flush(LOGDIR,DSP)
    lib.plot.tick()

    if iteration % 100 == 4:
        torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
        torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))
        generate_image()
    
    print('[iteration:'+str(iteration)+']')
