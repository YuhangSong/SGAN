'''machine'''
gan_ngpu = range(2)

'''expperiment'''
t = 3
lable = '1d_fall_r_a'
sess = 'grl1'
port = 10200

'''model'''
gan_size = 128
gan_nc = 3
state_depth = 1
gan_nz = 256
gan_aux_size = gan_nz/2
gan_batchsize = 128

# generate logdir according to config
logdir = '../../result/gmbrl_2/'+lable+'_t'+str(t)+'/'
modeldir = logdir+'model/'
datadir = logdir+'data/'

'''
    1d_fall:
        state determined by current action
                ..
                ....
        action  .......
                ....
                ..

'''
grid_type = '1d_fall' # 1d_fall
if grid_type is '1d_fall':
    action_space = gan_aux_size
elif grid_type is '2d_one_move' or grid_type is '2d_action_random' or grid_type is '2d_action':
    action_space = 4
grid_size = 8

'''behaviour'''
gan_worker_com_internal = 10
gan_save_image_internal = 60*1
gan_dataset_limit = 1024 * 2
gan_dataset_full_no_update = True
gan_recent_dataset = 10
gan_recent_recorder = 20
using_r = True
using_a = True


'''debug'''
gan_recent_dataset = 64
gan_worker_com_internal = 1
gan_save_image_internal = 0.0


'''waiting'''
overwirite_with_grid=True



