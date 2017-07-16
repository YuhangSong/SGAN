#!/usr/bin/env python
# coding=utf-8
import matplotlib as plt
import imageio
import numpy as np
import config
import cv2
import copy
import subprocess
file = 'A380.mp4'
dataset_file = 'dataset_4.npz'
dir = '../../dataset/'
filename = dir + file
vid = imageio.get_reader(filename,  'ffmpeg')
info = vid.get_meta_data()
num_frame = info['nframes']
size = info['size']
print info
num_step = int(info['duration']/config.gan_predict_interval)
frame_per_step = num_frame / num_step
last_image = None
dataset = []
for step in range(0, num_step):

    frame = step * frame_per_step

    image = np.asarray(vid.get_data(frame))/255.0

    c = []
    for color in range(3):
        temp = np.asarray(cv2.resize(image[:,:,color], (config.gan_size, config.gan_size)))
        temp = np.asarray(np.reshape(temp, (config.gan_size, config.gan_size, 1)))
        c += [copy.deepcopy(temp)]

    image = np.asarray(np.concatenate(c, axis=2))
    image = np.transpose(image, (2, 0, 1))

    print 'video>'+str(file)+'\t'+'step>'+str(step)+'\t'+'size>'+str(np.shape(image))

    if last_image is None:
        last_image = copy.deepcopy(image)
        continue
    else:
        pair = np.asarray([last_image,image])

        last_image = copy.deepcopy(image)

        dataset += [copy.deepcopy(np.asarray(pair))]

dataset = np.asarray(dataset)

print 'genrate dataset with size>'+str(np.shape(dataset))

np.savez(dir+dataset_file,
         dataset=dataset)

print(s)


