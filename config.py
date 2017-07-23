# exp time
t = 1
lable = 'd05_c05_auto_loss_fix_exp_simple_one_move'

# mode
run_on = 'agent' # agent, video

gan_size = 128
gan_nc = 3

if run_on is 'video':
    dataset_path = '../../dataset/'
    video_name_ = '3DPinball_1'
    video_name = video_name_+'.mp4'
    gan_predict_interval = 0.1
    dataset_name_ = video_name_+'_d'+str(gan_predict_interval).replace('.','')+'_c'+str(gan_size)+'_nc'+str(gan_nc)
    dataset_name = dataset_name_+'.npz'
elif run_on is 'agent':
    dataset_name_ = 'agent'

# gan model
gan_batchsize = 64
gan_nz = 256
gan_aux_size = gan_nz/2
gan_ngpu = 4
gan_dct = 4
gan_gctc = 4
gan_gctd = 4
gan_model_name_ = 'bs'+str(gan_batchsize)+'_nz'+str(gan_nz)+'_dct'+str(gan_dct)+'_gctc'+str(gan_gctc)+'_gctd'+str(gan_gctd)

# generate logdir according to config
logdir = '../../result/gmbrl_1/'+dataset_name_+'/'+gan_model_name_+'_l'+lable+'_t'+str(t)+'/'
modeldir = logdir+gan_model_name_+'/'
datadir = logdir+'data/'

if run_on is 'agent':
    """
    config rl env here
    """ 
    overwirite_with_grid = True
    grid_type = 'simple_one_move'
    action_space = 4
    grid_size = 8
    grid_target_x = 4
    grid_target_y = 4
    grid_action_random_discounter = 0.5
    gan_worker_com_internal = 10
    gan_save_image_internal = 60*1
    gan_dataset_limit = 1000
    gan_dataset_full_no_update = True
    '''since'''
    gan_recent_dataset = 10
    lower_gan_worker = 0.0
    lower_env_worker = 0.0
    agent_learning = False
    agent_acting = False

    gan_gloss_c_porpotion = 0.5

    # gan_recent_dataset = 64
    # gan_worker_com_internal = 1

