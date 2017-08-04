import torch
import torch.nn as nn
import torch.nn.parallel
import config

class DCGAN_D(nn.Module):

    '''
        deep conv GAN: D
    '''

    def __init__(self, isize, nz, nc, ngf, ngpu, depth, n_extra_layers=0):

        '''
            build the model
        '''

        # basic init
        super(DCGAN_D, self).__init__()
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

        # cat_layer
        self.cat_layer = torch.nn.Linear(nz+config.gan_aux_size, 1)

    def forward(self, input_image, input_aux):

        '''
            specific return when comute forward
        '''

        # compute output according to GPU parallel
        input_image = to_3d(input_image,self.nc)
        output_encoded = nn.parallel.data_parallel(self.main, input_image, self.ngpu)
        output_encoded = to_2d(output_encoded)

        cated = torch.cat([output_encoded, input_aux], 1)
        cated = torch.squeeze(cated,2)
        cated = torch.squeeze(cated,2)

        # compute output according to GPU parallel
        output = nn.parallel.data_parallel(self.cat_layer, cated, self.ngpu)

        # compute error
        error = output.mean(0)

        # return
        return error.view(1), output

class DCGAN_C(nn.Module):

    '''
        deep conv GAN: D
    '''

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):

        '''
            build the model
        '''

        # basic init
        super(DCGAN_C, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # starting main model
        main = nn.Sequential()

        # input is (4*nc) x isize x isize
        # the first (3*nc) channels are state, the 4th nc channel is prediction
        main.add_module('initial.conv_c.{0}-{1}'.format(4*nc, ndf),
                        nn.Conv2d(4*nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu_c.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # keep conv till
        while csize > config.gan_dct:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv_c'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm_c'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu_c'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size K x 4 x 4
        main.add_module('final.{0}-{1}.conv_c'.format(cndf, nz),
                        nn.Conv2d(cndf, nz, config.gan_dct, 1, 0, bias=False))

        # main model done
        self.main = main

        cat_layer = nn.Sequential()
        cat_layer.add_module('cat_linear',
                        nn.Linear(nz+config.gan_aux_size, 1))
        cat_layer.add_module('sigmoid_out',
                        nn.Sigmoid())
        self.cat_layer = cat_layer

    def forward(self, input_image, input_aux):

        '''
            specific return when comute forward
        '''

        input_image = to_3d(input_image,self.nc)
        # compute output according to GPU parallel
        output_encoded = nn.parallel.data_parallel(self.main, input_image, self.ngpu)
        output_encoded = to_2d(output_encoded)

        # cat them
        cated = torch.cat([output_encoded, input_aux], 1)
        cated = torch.squeeze(cated,2)
        cated = torch.squeeze(cated,2)

        # compute output according to GPU parallel
        output = nn.parallel.data_parallel(self.cat_layer, cated, self.ngpu)

        return output

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