# exp time
t = 1

# video
video_name_ = '3DPinball_1'
video_name = video_name_+'.mp4'

# dataset
gan_predict_interval = 0.1
gan_size = 128
gan_nc = 3
dataset_name_ = video_name_+'_d'+str(gan_predict_interval).replace('.','')+'_c'+str(gan_size)+'_nc'+str(gan_nc)
dataset_name = dataset_name_+'.npz'

# gan model
gan_batchsize = 64
gan_nz = 256
gan_ngpu = 2
gan_dct = 4
gan_gctc = 4
gan_gctd = 4
gan_model_name_ = 'bs'+str(gan_batchsize)+'_nz'+str(gan_nz)+'_dct'+str(gan_dct)+'_gctc'+str(gan_gctc)+'_gctd'+str(gan_gctd)+'_t'+str(t)

# generate logdir according to config
logdir = '../../result/gmbrl_1/'+dataset_name_+'/'+gan_model_name_+'/'
modeldir = logdir+gan_model_name_+'/'

# default
dataset_path = '../../dataset/'