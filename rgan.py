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

import visdom
vis = visdom.Visdom()

torch.manual_seed(1)

LOGDIR = '../../result/add_r_ro_gp_9/'
MODE = 'wgan-gp'  # wgan or wgan-gp
USE_R = True
DATASET = '2grid'  # 8gaussians, 25gaussians, swissroll, 2gaussians, 2grid
if DATASET is '2grid':
    GRID_SIZE = 8
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for
use_cuda = True
N_POINTS = 128

def prepare_dir():
    subprocess.call(["mkdir", "-p", LOGDIR])
    subprocess.call(["mkdir", "-p", LOGDIR+DATASET+'/'])
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

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2+2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2+2),
        )
        self.main = main

    def forward(self, noise_v, state_v):

        '''prepare'''
        state_v = state_v.squeeze(1)
        x = torch.cat([state_v,noise_v],1)

        '''forward'''
        x = self.main(x)

        '''decompose'''
        stater_v = x.narrow(1,0,2).unsqueeze(1)
        prediction_v = x.narrow(1,2,2).unsqueeze(1)
        x = torch.cat([stater_v,prediction_v],1)

        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2+2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, state_v, prediction_v):

        '''prepare'''
        state_v = state_v.squeeze(1)
        prediction_v = prediction_v.squeeze(1)
        x = torch.cat([state_v,prediction_v],1)

        '''forward'''
        x = self.main(x)
        x = x.view(-1)

        return x


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

frame_index = [0]

def generate_image():
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    plt.clf()

    '''get data'''
    data_fix_state = inf_train_gen(fix_state=True,fix_state_to=[5,4],batch_size=(N_POINTS**2))
    state_prediction_gt = torch.Tensor(data_fix_state.next()).cuda()
    state = state_prediction_gt.narrow(1,0,1)
    prediction_gt = state_prediction_gt.narrow(1,1,1)

    '''disc_map'''
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

    '''prediction_gt_samples'''
    samples = prediction_gt.squeeze(1).cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], c='orange', marker='+', alpha=0.5)


    '''prediction_gt_r_samples'''
    noise = torch.randn((BATCH_SIZE), 2).cuda()
    samples =   netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(prediction_gt, volatile=True)
                ).data.narrow(1,0,1).squeeze(1).cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', marker='+', alpha=0.5)

    '''prediction_samples'''
    noise = torch.randn((BATCH_SIZE), 2).cuda()
    samples =   netG(
                    noise_v = autograd.Variable(noise, volatile=True),
                    state_v = autograd.Variable(state, volatile=True)
                ).data.narrow(1,1,1).squeeze(1).cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+', alpha=0.5)


    plt.savefig(LOGDIR + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')
    plt_to_vis(plt.gcf(),'vis','fram_'+str(frame_index[0]))

    frame_index[0] += 1


# Dataset iterator
def inf_train_gen(fix_state=False, fix_state_to=[GRID_SIZE/2,GRID_SIZE/2], batch_size=BATCH_SIZE):

    if DATASET == '2grid':

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

                data = np.array([[cur_x,cur_y],
                                 [next_x,next_y]])
                dataset.append(data)
            dataset = np.array(dataset, dtype='float32')
            yield dataset


def calc_gradient_penalty(netD, state, prediction_gt, prediction):

    prediction_gt = prediction_gt.squeeze(1)
    prediction = prediction.squeeze(1)

    alpha = torch.rand(BATCH_SIZE, 1).cuda()
    alpha = alpha.expand(prediction_gt.size())

    interpolates = alpha * prediction_gt + ((1 - alpha) * prediction)

    interpolates = autograd.Variable(interpolates, requires_grad=True).unsqueeze(1)

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

data = inf_train_gen()

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

for iteration in xrange(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in xrange(CRITIC_ITERS):

        state_prediction_gt = torch.Tensor(data.next()).cuda()
        state = state_prediction_gt.narrow(1,0,1)
        prediction_gt = state_prediction_gt.narrow(1,1,1)

        if USE_R:
            noise = torch.randn(BATCH_SIZE, 2).cuda()
            prediction_gt = netG(
                                noise_v = autograd.Variable(noise, volatile=True),
                                state_v = autograd.Variable(prediction_gt, volatile=True)
                            ).data.narrow(1,0,1)
            state_prediction_gt = torch.cat([state,prediction_gt],1)

        netD.zero_grad()

        '''train with real'''
        D_real =    netD(
                        state_v = autograd.Variable(state),
                        prediction_v = autograd.Variable(prediction_gt)
                    ).mean()
        D_real.backward(mone)

        '''train with fake'''
        noise = torch.randn(BATCH_SIZE, 2).cuda()
        stater_prediction =  netG(
                                noise_v = autograd.Variable(noise, volatile=True),
                                state_v = autograd.Variable(state, volatile=True)
                            ).data
        prediction = stater_prediction.narrow(1,1,1)

        D_fake =    netD(
                        state_v = autograd.Variable(state),
                        prediction_v = autograd.Variable(prediction)
                    ).mean()
        D_fake.backward(one)

        '''train with gradient penalty'''
        gradient_penalty =  calc_gradient_penalty(
                                netD = netD,
                                state = state,
                                prediction_gt = prediction_gt,
                                prediction = prediction
                            )
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    ############################
    if USE_R:
        if Wasserstein_D.cpu().data.numpy()[0] > 0.0:
            update_type = 'g'
        else:
            update_type = 'r'
    else:
        update_type = 'g'
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation

    state_prediction_gt = torch.Tensor(data.next()).cuda()
    state = state_prediction_gt.narrow(1,0,1)
    prediction_gt = state_prediction_gt.narrow(1,1,1)

    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 2).cuda()
    stater_prediction_v = netG(
                            noise_v = autograd.Variable(noise),
                            state_v = autograd.Variable(state)
                        )
    stater_v = stater_prediction_v.narrow(1,0,1)
    prediction_v = stater_prediction_v.narrow(1,1,1)

    if update_type is 'g':
        G = netD(
                state_v = autograd.Variable(state),
                prediction_v = prediction_v
            ).mean()
        G_cost = -G
        lib.plot.plot(LOGDIR + DATASET + '/' + 'G_cost', G_cost.cpu().data.numpy())
        G.backward(mone)
    elif update_type is 'r':
        R = mse_loss_model(stater_v, autograd.Variable(state))
        R_cost = R
        lib.plot.plot(LOGDIR + DATASET + '/' + 'R_cost', R_cost.cpu().data.numpy())
        R.backward()
    
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(LOGDIR + DATASET + '/' + 'disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot(LOGDIR + DATASET + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
    

    if iteration % 100 == 99:
        lib.plot.flush()
        torch.save(netD.state_dict(), '{0}/netD.pth'.format(LOGDIR))
        torch.save(netG.state_dict(), '{0}/netG.pth'.format(LOGDIR))
        generate_image()

    lib.plot.tick()

    print('[iteration:'+str(iteration)+']')
