import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import config

class cat_layer(nn.Module):

    """docstring for rn_layer"""
    def __init__(self, conved_lenth, aux_lenth, encoded_lenth, output_lenth):
        super(cat_layer, self).__init__()

        # settings
        self.conved_lenth = conved_lenth
        self.aux_lenth = aux_lenth
        self.encoded_lenth = encoded_lenth
        self.output_lenth = output_lenth

        # NNs
        self.f_encoded = nn.Linear(self.conved_lenth, self.encoded_lenth).cuda()
        self.f_cat = nn.Linear(self.encoded_lenth+self.aux_lenth, self.output_lenth).cuda()

    def forward(self, x_aux, bloom_noise_offset=1):
        x_encoded = self.f_encoded(x_aux.narrow(1,0,self.conved_lenth))

        if x_aux.size()[1] > (self.conved_lenth+config.gan_aux_size):

            # there is noise, bloom
            slide_width = self.encoded_lenth+config.gan_aux_size+config.bloom_noise_lenth

            all_seq_cated_no_bloom_noise = torch.cat([x_encoded,x_aux.narrow(1,self.conved_lenth,int(config.gan_aux_size+config.fixed_noise_lenth))],1)

            bloom_noise_offset = bloom_noise_offset % slide_width

            bloom_noise = x_aux.narrow(1,self.conved_lenth+self.aux_lenth-config.bloom_noise_lenth,config.bloom_noise_lenth)

            before_bloom_noise = all_seq_cated_no_bloom_noise.narrow(1,0,bloom_noise_offset)
            after_bloom_noise = all_seq_cated_no_bloom_noise.narrow(1,bloom_noise_offset,(all_seq_cated_no_bloom_noise.size()[1]-bloom_noise_offset))

            to_cat_temp = [before_bloom_noise,bloom_noise,after_bloom_noise]

            final_cated = torch.cat(to_cat_temp, 1)

        else:

            # there isn't noise
            final_cated = torch.cat([x_encoded,x_aux.narrow(1,self.conved_lenth,self.aux_lenth)],1)
        
        x = self.f_cat(final_cated)

        return x

def compute_conv_parameters(num_channel_in, size_in, size_conved, channel_times, depth_in, depth_conved, direction_conv):

    # compute parameters
    num_layer = int(np.log2(size_in/size_conved))
    channel_i_dic = []
    channel_i_1_dic = []
    kernel_size_dic = []
    channel_i = channel_times / 2
    for layer in range(num_layer):
        if direction_conv:
            if (num_layer-layer) <= (depth_in-depth_conved):
                kernel_size = (2,4,4)
            else:
                kernel_size = (1,4,4)
        else:
            if layer < (depth_in-depth_conved):
                kernel_size = (2,4,4)
            else:
                kernel_size = (1,4,4)
        channel_i_1 = channel_i * 2
        if layer < 1:
            channel_i = num_channel_in
        channel_i_dic += [channel_i]
        channel_i_1_dic += [channel_i_1]
        kernel_size_dic += [kernel_size]
        channel_i = channel_i_1
    size_out = size_conved
    num_channel_out = channel_i_1

    return num_layer, size_out, num_channel_out, channel_i_dic, channel_i_1_dic, kernel_size_dic

class conv3d_layers(nn.Module):

    '''
        deep conv GAN: D
    '''

    def __init__(self, num_channel_in, size_in, size_conved, channel_times, depth_in, depth_conved):
        super(conv3d_layers, self).__init__()

        temp = compute_conv_parameters(num_channel_in, size_in, size_conved, channel_times, depth_in, depth_conved, True)
        self.num_layer, self.size_out, self.num_channel_out, self.channel_i_dic, self.channel_i_1_dic, self.kernel_size_dic = temp
    
        # NNs
        self.main = nn.Sequential()

        for layer in range(self.num_layer):

            self.main.add_module(   name='conv_{0}'.format(layer),
                                    module=nn.Conv3d(   in_channels=self.channel_i_dic[layer],
                                                        out_channels=self.channel_i_1_dic[layer],
                                                        kernel_size=self.kernel_size_dic[layer],
                                                        stride=(1,2,2),
                                                        padding=(0,1,1),
                                                        dilation=(1,1,1),
                                                        groups=1,
                                                        bias=False))
            if layer > 0:
                self.main.add_module(   name='batchnorm_{}'.format(layer),
                                        module=nn.BatchNorm3d(self.channel_i_1_dic[layer]))
            else:
                pass
            self.main.add_module(   name='relu_{}'.format(layer),
                                    module=nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.main(x)
        return x

class deconv3d_layers(nn.Module):

    def __init__(self, num_channel_in, size_in, size_conved, channel_times, depth_in, depth_conved):
        super(deconv3d_layers, self).__init__()

        temp = compute_conv_parameters(num_channel_in, size_in, size_conved, channel_times, depth_in, depth_conved, False)
        self.num_layer, self.size_out, self.num_channel_out, self.channel_i_dic, self.channel_i_1_dic, self.kernel_size_dic = temp
    
        # NNs
        self.main = nn.Sequential()

        # initail deconv
        self.main.add_module(   name='initial',
                                module=nn.ConvTranspose3d(  in_channels=config.gan_nz,
                                                            out_channels=self.channel_i_1_dic[self.num_layer-1],
                                                            kernel_size=(depth_conved,size_conved,size_conved),
                                                            stride=(1,1,1),
                                                            padding=(0,0,0),
                                                            dilation=(1,1,1),
                                                            bias=False))
        self.main.add_module(   name='batchnorm_initial',
                                module=nn.BatchNorm3d(self.channel_i_1_dic[self.num_layer-1]))
        self.main.add_module(   name='relu_inital',
                                module=nn.ReLU(True))

        for layer_ in range(self.num_layer):

            layer = self.num_layer - layer_ - 1

            self.main.add_module(   name='conv_{0}'.format(layer),
                                    module=nn.ConvTranspose3d(  in_channels=self.channel_i_1_dic[layer],
                                                                out_channels=self.channel_i_dic[layer],
                                                                kernel_size=self.kernel_size_dic[layer],
                                                                stride=(1,2,2),
                                                                padding=(0,1,1),
                                                                dilation=(1,1,1),
                                                                groups=1,
                                                                bias=False))

            if layer > 0:
                self.main.add_module(   name='batchnorm_{}'.format(layer),
                                        module=nn.BatchNorm3d(self.channel_i_dic[layer]))
                self.main.add_module(   name='relu_{}'.format(layer),
                                        module=nn.ReLU(True))
            else:
                self.main.add_module(   name='tanh_gd',
                                        module=nn.Tanh())

    def forward(self, x):
        x = self.main(x)
        return x

class DCGAN_D(nn.Module):

    def __init__(self):
        super(DCGAN_D, self).__init__()

        self.conv = conv3d_layers(  num_channel_in=config.gan_nc,
                                    size_in=config.gan_size,
                                    size_conved=config.gan_size_conved,
                                    channel_times=config.gan_channel_times,
                                    depth_in = (config.gan_state_lenth+1),
                                    depth_conved=1)

        # rn layer
        self.cat = cat_layer(   conved_lenth=self.conv.size_out*self.conv.size_out*self.conv.num_channel_out,
                                encoded_lenth=config.gan_nz,
                                aux_lenth=config.gan_aux_size*1,
                                output_lenth=1)

    def forward(self, input_image_v, input_aux_v):

        input_image_v = to_3d(input_image_v,config.gan_nc)

        # compute output according to GPU parallel
        if config.gan_ngpu > 1:
            conved_v = nn.parallel.data_parallel(self.conv, input_image_v, range(config.gan_ngpu))
        else: 
            conved_v = self.conv(input_image_v)

        conved_v = to_2d(conved_v)

        x = to_cat( input_conved_v=conved_v,
                    input_aux_v=input_aux_v,
                    input_noise_v=None)

        output = self.cat(x)

        # compute error
        error = output.mean(0)

        # return
        return error.view(1), output

class DCGAN_G(nn.Module):

    def __init__(self):
        super(DCGAN_G, self).__init__()

        self.conv = conv3d_layers(  num_channel_in=config.gan_nc,
                                    size_in=config.gan_size,
                                    size_conved=config.gan_size_conved,
                                    channel_times=config.gan_channel_times,
                                    depth_in = config.gan_state_lenth,
                                    depth_conved=1)

        # rn layer
        self.cat = cat_layer(   conved_lenth=self.conv.size_out*self.conv.size_out*self.conv.num_channel_out,
                                encoded_lenth=config.gan_nz,
                                aux_lenth=config.gan_aux_size*2,
                                output_lenth=config.gan_nz)

        self.deconv = deconv3d_layers(  num_channel_in=config.gan_nc,
                                        size_in=config.gan_size,
                                        size_conved=config.gan_size_conved,
                                        channel_times=config.gan_channel_times,
                                        depth_in = (config.gan_state_lenth+1),
                                        depth_conved=1)
        self.bloom_noise_offset = long(1)

    def bloom_noise(self):
        self.bloom_noise_offset += config.bloom_noise_step

    def forward(self, input_image_v, input_aux_v, input_noise_v):

        input_image_v = to_3d(input_image_v,config.gan_nc)

        if config.gan_ngpu > 1:
            conved_v = nn.parallel.data_parallel(self.conv, input_image_v, range(config.gan_ngpu))
        else: 
            conved_v = self.conv(input_image_v)

        conved_v = to_2d(conved_v)

        x = to_cat( input_conved_v=conved_v,
                    input_aux_v=input_aux_v,
                    input_noise_v=input_noise_v)

        x = self.cat(x,self.bloom_noise_offset)

        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze(x,2)

        if config.gan_ngpu > 1:
            x = nn.parallel.data_parallel(self.deconv, x, range(config.gan_ngpu))
        else: 
            x = self.deconv(x)

        x = to_2d(x)

        return x

def to_cat(input_conved_v, input_aux_v, input_noise_v=None):

    concated = []

    input_conved_v = input_conved_v.view(input_conved_v.size()[0],input_conved_v.size()[1]*input_conved_v.size()[2]*input_conved_v.size()[3])
    concated += [input_conved_v]

    concated += [input_aux_v]

    if input_noise_v is not None:
        concated += [input_noise_v]

    conved_v_noise_v_action_v = torch.cat(concated,1)

    return conved_v_noise_v_action_v

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
