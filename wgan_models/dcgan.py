import torch
import torch.nn as nn
import torch.nn.parallel
import config

class DCGAN_D(nn.Module):

    def __init__(self):
        super(DCGAN_D, self).__init__()

        self.conv_layer = torch.nn.Linear(config.action_space*(config.state_depth+1), config.gan_nz)

        self.cat_layer = torch.nn.Linear(config.gan_nz+config.gan_aux_size, 1)

    def forward(self, input_image, input_aux):

        input_image = input_image.contiguous()
        input_image = input_image.view(input_image.size()[0],-1)

        encoded = nn.parallel.data_parallel(    self.conv_layer, 
                                                input_image,
                                                config.gan_ngpu)

        x = nn.parallel.data_parallel(          self.cat_layer,
                                                torch.cat([encoded,input_aux],1),
                                                config.gan_ngpu)

        return x

class DCGAN_G(nn.Module):

    def __init__(self):
        super(DCGAN_G, self).__init__()

        self.conv_layer     = torch.nn.Linear(config.state_depth*config.action_space,                config.gan_nz)

        self.cat_layer      = torch.nn.Linear(config.gan_nz+2*config.gan_aux_size,  config.gan_nz)

        self.deconv_layer   = nn.Sequential()
        self.deconv_layer.add_module(   'deconv.Linear',
                                        nn.Linear(config.gan_nz,(config.state_depth+1)*config.action_space))
        self.deconv_layer.add_module(   'final.Sigmoid',
                                        nn.Sigmoid())

    def forward(self, input_image, input_aux, input_noise):
        input_image = input_image.contiguous()
        input_image = input_image.view(input_image.size()[0],-1)

        encoded = nn.parallel.data_parallel(    self.conv_layer, 
                                                input_image, 
                                                config.gan_ngpu)

        x       = nn.parallel.data_parallel(    self.cat_layer,
                                                torch.cat([encoded,input_aux,input_noise],1),
                                                config.gan_ngpu)

        x       = nn.parallel.data_parallel(    self.deconv_layer, 
                                                x, 
                                                config.gan_ngpu)

        x = x.view(x.size()[0],(config.state_depth+1),config.action_space)

        return x

class DCGAN_G_Cv(nn.Module):

    '''
        deep conv GAN: G
    '''

    def __init__(self, isize, nz, nc, ngf, ngpu, depth, n_extra_layers=0):

        '''
            build model
        '''

        # basic intialize
        super(DCGAN_G_Cv, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # starting main model
        main = nn.Sequential()

        # input is (3*nc) x isize x isize
        # which is 3 sequential states
        # initial conv
        main.add_module('initial.conv_gc.{0}-{1}'.format(nc, ngf),
                                               nn.Conv3d(nc, ngf, (2,4,4), (1,2,2), (0,1,1), bias=False))
        main.add_module('initial.relu_gc.{0}'.format(ngf),
                                                       nn.LeakyReLU(0.2, inplace=True))
        csize, cndf, depth = isize / 2, ngf, depth - 1

        # conv till
        while csize > config.gan_gctc:
            in_feat = cndf
            out_feat = cndf * 2
            if depth > 1:
                kernel_size = (2,4,4)
            else:
                kernel_size = (1,4,4)
            main.add_module('pyramid.{0}-{1}.conv_gc'.format(in_feat, out_feat),
                            nn.Conv3d(in_feat, out_feat, kernel_size, (1,2,2), (0,1,1), bias=False))
            main.add_module('pyramid.{0}.batchnorm_gc'.format(out_feat),
                            nn.BatchNorm3d(out_feat))
            main.add_module('pyramid.{0}.relu_gc'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2  
            csize = csize / 2
            depth = depth - 1

        # conv final to nz
        # state size. K x 1 x 4 x 4
        main.add_module('final.{0}-{1}.conv_gc'.format(cndf, nz),
                        nn.Conv3d(cndf, nz, (1,csize,csize), (1,1,1), (0,0,0), bias=False))

        # main model done
        self.main = main

    def forward(self, input):

        '''
            specific forward comute
        '''
        input = to_3d(input,self.nc)
        # compute output according to gpu parallel
        output = nn.parallel.data_parallel(self.main, input, self.ngpu)
        output = to_2d(output)

        # return
        return output

class DCGAN_G_DeCv(nn.Module):

    '''
        deep conv GAN: G
    '''

    def __init__(self, isize, nz, nc, ngf, ngpu, depth_out, n_extra_layers=0):

        '''
            build model
        '''

        # basic intialize
        super(DCGAN_G_DeCv, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # starting main model
        main = nn.Sequential()

        # compute initial cngf for deconv
        cngf, tisize = ngf//2, config.gan_gctd
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        # initail deconv
        main.add_module('initial.{0}-{1}.conv_gd'.format(nz*2, cngf),
                        nn.ConvTranspose3d(nz*2, cngf, (1,config.gan_gctd,config.gan_gctd), (1,1,1), (0,0,0), bias=False))
        main.add_module('initial.{0}.batchnorm_gd'.format(cngf),
                        nn.BatchNorm3d(cngf))
        main.add_module('initial.{0}.relu_gd'.format(cngf),
                        nn.ReLU(True))
        csize, cndf, depth = config.gan_gctd, cngf, 1

        # deconv till
        while csize < isize//2:
            if depth < depth_out:
                kernel_size = (2,4,4)
            else:
                kernel_size = (1,4,4)
            main.add_module('pyramid.{0}-{1}.conv_gd'.format(cngf, cngf//2),
                            nn.ConvTranspose3d(cngf, cngf//2, kernel_size, (1,2,2), (0,1,1), bias=False))
            main.add_module('pyramid.{0}.batchnorm_gd'.format(cngf//2),
                            nn.BatchNorm3d(cngf//2))
            main.add_module('pyramid.{0}.relu_gd'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2
            depth = depth + 1

        # # layer for final output
        main.add_module('final.{0}-{1}.conv_gd'.format(cngf, nc),
                        nn.ConvTranspose3d(cngf, nc, (1,4,4), (1,2,2), (0,1,1), bias=False))
        main.add_module('final.{0}.tanh_gd'.format(nc),
                        nn.Tanh())

        # main model done
        self.main = main

    def forward(self, input):

        '''
            specific forward comute
        '''
        input = input.unsqueeze(2)
        # compute output according to gpu parallel
        output = nn.parallel.data_parallel(self.main, input, self.ngpu)
        output = to_2d(output)

        # return
        return output

def to_3d(x,nc):
    depth = x.size()[1]/nc
    x = torch.unsqueeze(x,1)
    one_depth = []
    for channel_i in range(nc):
        one_channel = []
        for depth_i in range(depth):
            one_channel += [x.narrow(2,(depth_i*nc+channel_i),1)]
        one_depth += [torch.cat(one_channel,2)]
    x = torch.cat(one_depth,1)
    return x

def to_2d(x):
    nc = x.size()[1]
    depth = x.size()[2]
    one_image = []
    for channel_i in range(nc):
        for depth_i in range(depth):

            one_image += [x.narrow(1,channel_i,1).narrow(2,depth_i,1)]
    x = torch.cat(one_image,2)
    x = torch.squeeze(x,1)
    return x