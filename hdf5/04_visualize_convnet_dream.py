#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)

import keras
print (keras.__version__)

import matplotlib.pyplot as plt
import numpy as np

import os

#%%Load model
from keras.models import load_model
model_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#model_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396.h5'
model = load_model(model_to_load)
model.summary()
input_shape = (1,) + model.input_shape[1:]
#%%create model that compute activation output
from support import ids
from keras import backend as K
#layer_name = 'conv2d_2'
#filter_index = 0
#
#layer_output = model.get_layer(layer_name).output
#loss = K.mean(layer_output[:, :, :, filter_index])
#
## The call to `gradients` returns a list of tensors (of size 1 in this case)
## hence we only keep the first element -- which is a tensor.
#grads = K.gradients(loss, model.input)[0]
#
## We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
#iterate = K.function([model.input], [loss, grads])
#
## Let's test it:
#loss_value, grads_value = iterate([np.zeros(input_shape)])
#
## We start from a gray image with some noise
##input_img_data = np.random.random(input_shape) * 20 + 128.
#input_img_data = np.random.random(input_shape) * 2  - 1. #generate value [-1, 1]
#
## Run gradient ascent for 40 steps
#step = 1.  # this is the magnitude of each gradient update
#for i in range(40):
#    # Compute the loss value and gradient value
#    loss_value, grads_value = iterate([input_img_data])
#    # Here we adjust the input image in the direction that maximizes the loss
#    input_img_data += grads_value * step
#    
##%%
#plt.imshow(ids.generate_pattern(model,layer_name, 0,input_shape))
#plt.show()   

#%%
layerToShow = ['conv2d_1', 'conv2d_2', 'conv2d_3'] 
#layerToShow = ['activation_1', 'activation_1', 'activation_3'] 
max_depth = [ model.get_layer(layer_name).output.shape[-1].value for layer_name in layerToShow]
max_depth = np.max(max_depth)
tiles = int(np.ceil(np.sqrt(max_depth)))
size = input_shape[1]
margin = 5
for layer_name in layerToShow:


    # This a empty (black) image where we will store our results.
    results = np.zeros((tiles * size + (tiles-1) * margin, tiles * size + (tiles-1) * margin, input_shape[3]))

    for i in range(tiles):  # iterate over the rows of our results grid
        for j in range(tiles):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            if i + (j * tiles) >=  model.get_layer(layer_name).output.shape[-1].value:
                continue
            filter_img = ids.generate_pattern(model,layer_name, i + (j * tiles), input_shape)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()