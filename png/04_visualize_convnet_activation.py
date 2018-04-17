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
#model_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#model_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396.h5'
model_to_load = 'ids_Cscan0e01sgd1522074767-05-0.94.hdf5'
model = load_model(model_to_load)
model.summary()

#%%
from support import ids_dataset
from support.evaluation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from support import ids
#scan_type = 'B'
#scan_type = 'T'
scan_type = 'C'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_val.txt'
(x_val_c,y_val_c),(anom_id,acq_id)=ids.get_central_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale)

filename = scan_type+'_sublist_train_mean.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist)

x_val_c = (x_val_c - x_mu)/255

#%% load image
from keras.preprocessing import image
import re
is_target_id_re = re.compile(r"T18")
#is_target_id_re = re.compile(r"FA21")
img = x_val_c[0]
for i,anom in enumerate(anom_id):
    is_target_id_res = is_target_id_re.search(anom)
    if is_target_id_res != None:
        img = x_val_c[i]
        print('image {} found!'.format(anom))
        break
img_c = np.squeeze(img[:,:,1])
#img_c = np.expand_dims(img_c,axis=2)
plt.figure()
plt.matshow(np.squeeze(img[:,:,1]),cmap='gray')
#plt.imshow(image.array_to_img(img_c))
plt.show()
#%%create model that compute activation output
from keras import models

#levelToShow = 6
levelToShow = 9
#levelToShow = 11
# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:levelToShow]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(np.expand_dims(img,axis=0))
#%% Plot
# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:levelToShow]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
#    plt.imshow(display_grid, aspect='auto', cmap='gray')
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()






