#!/usr/bin/env python
# coding=utf-8
import matplotlib as plt
import imageio
import numpy as np
import config
import cv2
import copy
import subprocess
file = config.video_name
dataset_file = config.dataset_name
dir = config.dataset_path
filename = dir + file
vid = imageio.get_reader(filename,  'ffmpeg')
info = vid.get_meta_data()
num_frame = info['nframes']
size = info['size']
print info
num_step = int(info['duration']/config.gan_predict_interval)
frame_per_step = num_frame / num_step
lllast_image = None
llast_image = None
last_image = None
dataset = []
for step in range(0, num_step):

    frame = step * frame_per_step

    image = np.asarray(vid.get_data(frame))/255.0

    c = []
    for color in range(3):
        temp = np.asarray(cv2.resize(image[:,:,color], (config.gan_size, config.gan_size)))
        temp = np.expand_dims(temp,0)
        c += [copy.deepcopy(temp)]

    if config.gan_nc is 1:
        image = np.asarray(np.add(c[0]*0.299,c[1]*0.587))
        image = np.asarray(np.add(image,c[2]*0.114))
    elif config.gan_nc is 3:
        image = np.concatenate((c[0],c[1],c[2]),0)

    print 'video>'+str(file)+'\t'+'step>'+str(step)+'\t'+'size>'+str(np.shape(image))
    
    if last_image is None or llast_image is None or lllast_image is None:
        pass
    else:
        pair = [lllast_image,llast_image,last_image,image]
        dataset += [copy.deepcopy(np.asarray(pair))]

    lllast_image = copy.deepcopy(llast_image)
    llast_image = copy.deepcopy(last_image)
    last_image = copy.deepcopy(image)

dataset = np.asarray(dataset)

print 'genrate dataset with size>'+str(np.shape(dataset))

np.savez(dir+dataset_file,
         dataset=dataset)

print(s)


