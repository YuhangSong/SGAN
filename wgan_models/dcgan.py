import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import config

class rn_layer(nn.Module):

    """docstring for rn_layer"""
    def __init__(self, num, lenth, aux_lenth=0, size=256, output_size=256):
        super(rn_layer, self).__init__()

        # settings
        self.num = num
        self.lenth = lenth
        self.aux_lenth = aux_lenth
        self.size = size
        self.output_size = output_size

        # NNs
        self.g_fc1 = nn.Linear((self.lenth+1)*2+self.aux_lenth, self.size).cuda()
        self.g_fc2 = nn.Linear(self.size, self.size).cuda()
        self.g_fc3 = nn.Linear(self.size, self.size).cuda()
        self.g_fc4 = nn.Linear(self.size, self.output_size).cuda()
        self.f_fc1 = nn.Linear(self.size, self.size).cuda()
        self.f_fc2 = nn.Linear(self.size, self.size).cuda()
        self.f_fc3 = nn.Linear(self.size, self.output_size).cuda()

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(config.gan_batchsize, num, 1).cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((config.gan_batchsize, self.num, 1))
        for i in range(self.num):
            np_coord_tensor[:,i,:] = np.array([i])
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

    def forward(self, x_aux):

        # 0: batch
        # 1: number
        # 2: feature

        batch_size = x_aux.size()[0]

        # coordinate tensor is cutted to fit in a
        # smaller batch resulted from gpu para
        coord_tensor_batch = self.coord_tensor.narrow(0,0,batch_size)

        x = x_aux.narrow(2,0,self.lenth)
        x = torch.cat([x, coord_tensor_batch],2)

        # add coordinates
        x_aux = torch.cat([x_aux, coord_tensor_batch],2)

        # cast all pairs against each other i
        x_i = torch.unsqueeze(x,1)
        x_i = x_i.repeat(1,self.num,1,1)

        # cast all pairs against each other j
        x_j = torch.unsqueeze(x_aux,2)
        x_j = x_j.repeat(1,1,self.num,1)

        # concatenate all together
        x_full = torch.cat([x_i,x_j],3)

        # reshape for passing through network
        x_ = x_full.view(x_full.size()[0]*x_full.size()[1]*x_full.size()[2],x_full.size()[3])

        x_ = self.g_fc1(x_)        
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(batch_size,x_.size()[0]/batch_size,self.output_size)
        x_g = x_g.sum(1).squeeze()

        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f)
        x_f = self.f_fc3(x_f)

        return x_f

class DCGAN_D(nn.Module):

    '''
        deep conv GAN: D
    '''

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):

        '''
            build the model
        '''

        # basic init
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # starting main model
        main = nn.Sequential()

        # input is (4*nc) x isize x isize
        # the first (3*nc) channels are state, the 4th nc channel is prediction
        main.add_module('initial.conv_d.{0}-{1}'.format(4*nc, ndf),
                        nn.Conv2d(4*nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu_d.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # keep conv till
        while csize > config.gan_dct:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv_d'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm_d'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu_d'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size K x 4 x 4
        main.add_module('final.{0}-{1}.conv_d'.format(cndf, nz),
                        nn.Conv2d(cndf, nz, config.gan_dct, 1, 0, bias=False))

        # main model done
        self.main = main

        # cat_layer
        self.cat_layer = torch.nn.Linear(nz+config.gan_aux_size, 1)

    def forward(self, input_image, input_aux):

        '''
            specific return when comute forward
        '''

        # compute output according to GPU parallel
        if isinstance(input_image.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output_encoded = nn.parallel.data_parallel(self.main, input_image, range(self.ngpu))
        else: 
            output_encoded = self.main(input_image)

        cated = torch.cat([output_encoded, input_aux], 1)
        cated = torch.squeeze(cated,2)
        cated = torch.squeeze(cated,2)

        # compute output according to GPU parallel
        if isinstance(cated.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.cat_layer, cated, range(self.ngpu))
        else: 
            output = self.cat_layer(cated)

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

        # compute output according to GPU parallel
        if isinstance(input_image.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output_encoded = nn.parallel.data_parallel(self.main, input_image, range(self.ngpu))
        else: 
            output_encoded = self.main(input_image)

        # cat them
        cated = torch.cat([output_encoded, input_aux], 1)
        cated = torch.squeeze(cated,2)
        cated = torch.squeeze(cated,2)

        # compute output according to GPU parallel
        if isinstance(cated.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.cat_layer, cated, range(self.ngpu))
        else: 
            output = self.cat_layer(cated)

        return output

class DCGAN_G_Cv(nn.Module):

    '''
        deep conv GAN: G
    '''

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):

        '''
            build model
        '''

        # basic intialize
        super(DCGAN_G_Cv, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # starting main model
        main = nn.Sequential()

        # input is (3*nc) x isize x isize
        # which is 3 sequential states
        # initial conv
        main.add_module('initial.conv_gc.{0}-{1}'.format(3*nc, ngf),
                                               nn.Conv2d(3*nc, ngf, 4, 2, 1, bias=False))
        main.add_module('initial.relu_gc.{0}'.format(ngf),
                                                       nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ngf

        # conv till
        while csize > config.gan_gctc:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv_gc'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm_gc'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu_gc'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2  
            csize = csize / 2

        # # conv final to nz
        # # state size. K x 4 x 4
        # main.add_module('final.{0}-{1}.conv_gc'.format(cndf, nz),
        #                 nn.Conv2d(cndf, nz, config.gan_gctc, 1, 0, bias=False))

        # main model done
        self.main = main

    def forward(self, input):

        '''
            specific forward comute
        '''

        # compute output according to gpu parallel
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)

        # return
        return output

class DCGAN_G_DeCv(nn.Module):

    '''
        deep conv GAN: G
    '''

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):

        '''
            build model
        '''

        # basic intialize
        super(DCGAN_G_DeCv, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # rn layer
        self.rn = rn_layer(num=config.gan_gctc*config.gan_gctc,
                           lenth=512,
                           aux_lenth=config.gan_aux_size*2,
                           size=nz,
                           output_size=nz)

        # starting main model
        main = nn.Sequential()

        # compute initial cngf for deconv
        cngf, tisize = ngf//2, config.gan_gctd
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        # initail deconv
        main.add_module('initial.{0}-{1}.conv_gd'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, config.gan_gctd, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm_gd'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu_gd'.format(cngf),
                        nn.ReLU(True))
        csize, cndf = config.gan_gctd, cngf

        # deconv till
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.conv_gd'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm_gd'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu_gd'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # layer for final output
        main.add_module('final.{0}-{1}.conv_gd'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh_gd'.format(nc),
                        nn.Tanh())

        # main model done
        self.main = main

    def forward(self, x):

        '''
            specific forward comute
        '''

        # compute output according to gpu parallel
        if self.ngpu > 1:
            x = nn.parallel.data_parallel(self.rn, x, range(self.ngpu))
        else: 
            x = self.rn(x)

        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze(x,2)

        # compute output according to gpu parallel
        if self.ngpu > 1:
            x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else: 
            x = self.main(x)

        # return
        return x
