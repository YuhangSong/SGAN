# exp time
t=8
lable = 'd05_c05_auto_loss_fix_exp_simple_one_move_fix_auto_dc_f2_dg_g_ruiner_normal_game_3dcnn_pre_ruiner_keep_mse_exp_c_onehot_niv_removed_in_lgd'

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
gan_ngpu = 2
gan_batchsize = 64 * gan_ngpu
gan_nz = 256
gan_aux_size = gan_nz/2

gan_dct = 4
gan_gctc = gan_dct
gan_gctd = gan_gctc
gan_size_conved = gan_dct
gan_channel_times = 32

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
    grid_type = 'normal' # simple_one_move, normal
    grid_random_type = 'russell' # discount
    action_space = 4
    grid_size = 5
    grid_target_x = 4
    grid_target_y = 4
    grid_action_random_discounter = 0.5
    gan_worker_com_internal = 10
    gan_save_image_internal = 60*5
    gan_dataset_limit = 1024*2
    gan_dataset_full_no_update = False
    gan_recent_dataset = 10
    lower_gan_worker = 0.0
    lower_env_worker = 0.0
    agent_learning = False
    agent_acting = False
    train_corrector = False
    DCiters_ = 5
    noise_image = 0.2
    ruiner_train_to_mse = 0.001
    loss_g_factor = 2.0
    bloom_noise_rate = 0.5
    bloom_noise_lenth = int(gan_aux_size*bloom_noise_rate)
    fixed_noise_lenth = gan_aux_size-bloom_noise_lenth
    bloom_noise_step = int(bloom_noise_lenth*0.0) # max to the effect of bloom_noise_rate, zero to no bloom
    bloom_at_errD = 0.25
    niv_rate = 1.0
    donot_niv_gate = 0.2
    do_niv_p_gate = 0.2

    gan_gloss_c_porpotion = 0.0
    auto_d_c_factor = 2

    # to fasten training, only use when debug
    # gan_recent_dataset = gan_batchsize
    # gan_worker_com_internal = 10
    # gan_save_image_internal = 5
    # ruiner_train_to_mse = 0.8
    # bloom_at_errD = 4.0

