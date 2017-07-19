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
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, num_local_steps, visualise, gan_runner):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.gan_runner = gan_runner

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise, self.gan_runner)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

class GanRunnerThread(threading.Thread):
    """
    This thread runs gan training
    """
    def __init__(self):
        threading.Thread.__init__(self)
        
        '''config'''
        self.cuda = True
        self.ngpu = config.gan_ngpu
        self.nz = config.gan_nz
        self.ngf = 64
        self.ndf = 64
        self.nc = config.gan_nc
        self.n_extra_layers = 0
        self.imageSize = config.gan_size
        self.lrD = 0.00005
        self.lrG = 0.00005
        self.batchSize = config.gan_batchsize
        self.Diters_ = 5
        self.clamp_lower = -0.01
        self.clamp_upper = 0.01
        self.experiment = config.logdir
        self.dataset_limit = 500

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
        self.netG_Cv = dcgan.DCGAN_G_Cv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netG_DeCv = dcgan.DCGAN_G_DeCv(self.imageSize, self.nz, self.nc, self.ngf, self.ngpu, self.n_extra_layers)
        self.netD = dcgan.DCGAN_D(self.imageSize, self.nz, self.nc, self.ndf, self.ngpu, self.n_extra_layers)

        '''init models'''
        self.netG_Cv.apply(weights_init)
        self.netG_DeCv.apply(weights_init)
        self.netD.apply(weights_init)

        '''do auto checkpoint'''
        try:
            self.netG_Cv.load_state_dict(torch.load(config.modeldir+'netG_Cv.pth'))
            print('Previous checkpoint for netG_Cv founded')
        except Exception, e:
            print('Previous checkpoint for netG_Cv unfounded')
        try:
            self.netG_DeCv.load_state_dict(torch.load(config.modeldir+'netG_DeCv.pth'))
            print('Previous checkpoint for netG_DeCv founded')
        except Exception, e:
            print('Previous checkpoint for netG_DeCv unfounded')
        try:
            self.netD.load_state_dict(torch.load(config.modeldir+'netD.pth'))
            print('Previous checkpoint for netD founded')
        except Exception, e:
            print('Previous checkpoint for netD unfounded')

        '''print the models'''
        print(self.netG_Cv)
        print(self.netG_DeCv)
        print(self.netD)

        self.inputd = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputd_real_part = torch.FloatTensor(self.batchSize, 4, self.imageSize, self.imageSize)
        self.inputg = torch.FloatTensor(self.batchSize, 3, self.imageSize, self.imageSize)
        self.noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1)
        self.fixed_noise = torch.FloatTensor(self.batchSize, self.nz, 1, 1).normal_(0, 1)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        '''dataset intialize'''
        self.dataset = torch.FloatTensor(np.zeros((1, 4, self.nc, self.imageSize, self.imageSize)))
        self.dataset_sampler_indexs = torch.LongTensor(self.batchSize)

        '''convert tesors to cuda type'''
        if self.cuda:
            self.netD.cuda()
            self.netG_Cv.cuda()
            self.netG_DeCv.cuda()
            self.inputd = self.inputd.cuda()
            self.inputg = self.inputg.cuda()
            self.one, self.mone = self.one.cuda(), self.mone.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()
            # self.dataset = self.dataset.cuda()
            # self.dataset_sampler_indexs = self.dataset_sampler_indexs.cuda()

        '''create optimizer'''
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
        self.optimizerG_Cv = optim.RMSprop(self.netG_Cv.parameters(), lr = self.lrG)
        self.optimizerG_DeCv = optim.RMSprop(self.netG_DeCv.parameters(), lr = self.lrG)

        self.iteration_i = 0

    def push_data(self, data):
        data = torch.FloatTensor(data)

        self.dataset = torch.cat(seq=[self.dataset,data],
                                 dim=0)

        if self.dataset.size()[0] > self.dataset_limit:
            self.dataset = self.dataset.narrow(dimension=0,
                                               start=self.dataset.size()[0]-self.dataset_limit,
                                               length=self.dataset_limit)

    def run(self):

        while True:
            while self.dataset.size()[0] >= self.batchSize:
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
                if self.iteration_i < 25 or self.iteration_i % 500 == 0:
                    Diters = 100
                else:
                    Diters = self.Diters_

                '''
                    start interation training of D network
                    D network is trained for sevrel steps when 
                    G network is trained for one time
                '''
                j = 0
                while j < Diters:
                    j += 1

                    # clamp parameters to a cube
                    for p in self.netD.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    ######## train D network with real #######

                    ## random sample from dataset ##
                    raw = torch.index_select(self.dataset,0,self.dataset_sampler_indexs.random_(0,self.dataset.size()[0]))
                    image = [] 
                    for image_i in range(4):
                        image += [raw.narrow(1,image_i,1)]
                    state_prediction_gt = torch.cat(image,2)
                    state_prediction_gt = torch.squeeze(state_prediction_gt,1)
                    if self.cuda:
                        state_prediction_gt = state_prediction_gt.cuda()
                    state = state_prediction_gt.narrow(1,0*self.nc,3*self.nc)
                    prediction_gt = state_prediction_gt.narrow(1,3*self.nc,1*self.nc)

                    ######### train D with real ########

                    # reset grandient
                    self.netD.zero_grad()

                    # feed
                    self.inputd.resize_as_(state_prediction_gt).copy_(state_prediction_gt)
                    inputdv = Variable(self.inputd)

                    # compute
                    errD_real, outputD_real = self.netD(inputdv)
                    errD_real.backward(self.one)

                    ########### get fake #############

                    # feed
                    self.inputg.resize_as_(state).copy_(state)
                    inputgv = Variable(self.inputg, volatile = True) # totally freeze netG

                    # compute encoded
                    encodedv = self.netG_Cv(inputgv)

                    # compute noise
                    self.noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
                    noisev = Variable(self.noise, volatile = True) # totally freeze netG

                    # concate encodedv and noisev
                    encodedv_noisev = torch.cat([encodedv,noisev],1)

                    # predict
                    prediction = self.netG_DeCv(encodedv_noisev)
                    prediction = prediction.data
                    
                    ############ train D with fake ###########

                    # get state_prediction
                    state_prediction = torch.cat([state, prediction], 1)

                    # feed
                    self.inputd.resize_as_(state_prediction).copy_(state_prediction)
                    inputdv = Variable(self.inputd)

                    # compute
                    errD_fake, outputD_fake = self.netD(inputdv)
                    errD_fake.backward(self.mone)

                    # optmize
                    errD = errD_real - errD_fake
                    self.optimizerD.step()

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

                # reset grandient
                self.netG_Cv.zero_grad()
                self.netG_DeCv.zero_grad()

                # feed
                self.inputg.resize_as_(state).copy_(state)
                inputgv = Variable(self.inputg)

                # compute encodedv
                encodedv = self.netG_Cv(inputgv)

                # compute noisev
                self.noise.resize_(self.batchSize, self.nz, 1, 1).normal_(0, 1)
                noisev = Variable(self.noise)

                # concate encodedv and noisev
                encodedv_noisev = torch.cat([encodedv,noisev],1)

                # predict
                prediction = self.netG_DeCv(encodedv_noisev)

                # get state_predictionv, this is a Variable cat 
                statev_predictionv = torch.cat([Variable(state), prediction], 1)

                # feed, this state_predictionv is Variable
                inputdv = statev_predictionv

                # compute
                errG, _ = self.netD(inputdv)
                errG.backward(self.one)

                # optmize
                self.optimizerG_Cv.step()
                self.optimizerG_DeCv.step()

                ######################################################################
                ###################### End of Update G network #######################
                ######################################################################


                ######################################################################
                ########################### One Iteration ### ########################
                ######################################################################

                '''log result'''
                print('[iteration_i:%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                    % (self.iteration_i,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

                '''log image result'''
                if self.iteration_i % 100 == 0:

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
                    vutils.save_image(sample2image(state_prediction_gt[0]), '{0}/real_samples_{1}.png'.format(self.experiment, self.iteration_i))

                    '''log perdict result'''
                    vutils.save_image(sample2image(state_prediction[0]), '{0}/fake_samples_{1}.png'.format(self.experiment, self.iteration_i))

                    '''do checkpointing'''
                    torch.save(self.netG_Cv.state_dict(), '{0}/{1}/netG_Cv.pth'.format(self.experiment,config.gan_model_name_))
                    torch.save(self.netG_DeCv.state_dict(), '{0}/{1}/netG_DeCv.pth'.format(self.experiment,config.gan_model_name_))
                    torch.save(self.netD.state_dict(), '{0}/{1}/netD.pth'.format(self.experiment,config.gan_model_name_))

                self.iteration_i += 1
                ######################################################################
                ######################### End One in Iteration  ######################
                ######################################################################

def rbg2gray(rgb):
    gray = rgb[0]*0.299 + rgb[1]*0.587 + rgb[2]*0.114  # Gray = R*0.299 + G*0.587 + B*0.114
    gray = np.expand_dims(gray,2)
    return gray

def env_runner(env, policy, num_local_steps, summary_writer, render, gan_runner):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """

    '''create image recorder'''
    lllast_image = None
    llast_image = None
    last_image = None
    image = None

    last_image = env.reset()
    last_state = rbg2gray(last_image)
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            if config.overwirite_with_grid:
                GlobalVar.set_mq_client(action.argmax())
                
            # argmax to convert from one-hot
            image, reward, terminal, info = env.step(action.argmax())

            if last_image is None or llast_image is None or lllast_image is None:
                pass
            else:
                data = [lllast_image,llast_image,last_image,image]
                data = np.asarray(data)
                gan_runner.push_data(np.expand_dims(data,0))

            lllast_image = copy.deepcopy(llast_image)
            llast_image = copy.deepcopy(last_image)
            last_image = copy.deepcopy(image)

            
            state = rbg2gray(image)

            # gan_runner.push_data(state_rgb)

            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
            
            if terminal:
                terminal_end = True
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                '''reset image recorder'''
                lllast_image = None
                llast_image = None
                last_image = None
                image = None
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, env, task, visualise):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """

        self.env = env
        self.task = task

        '''create gan_runner'''
        self.gan_runner = GanRunnerThread()

        ######################################################################
        ############################## A3C Model #############################
        ######################################################################

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20, visualise, self.gan_runner)


            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

        ######################################################################

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.gan_runner.start()
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
