# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)
sys.path.append('/home/lfabbrini/spyder/keras_wdir')

import keras
print (keras.__version__)
from keras.preprocessing import image #img_to_array
import matplotlib.pyplot as plt
import numpy as np

import os
#import re
from time import time
from support import ids_dataset
from support import ids

scan_type = 'C'
#scan_type = 'T'
#scan_type = 'B'

data_type = 'V'
#path_data = '/home/lfabbrini/data'
path_data = '/media/sf_share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_20184517447/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
model_dir = 'mdl_tr80val10te10fs1_tex1_0001'
#model_dir = 'mdl_debug'
grayscale=False
hdf5_format=True
max_value = 1

filename = data_type+'_sublist_train.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train,y_train) = ids_dataset.load_data(filelist,hdf5_format) #x_train (N,CH,X,Z) 
#downsampling by 2
#x_train = x_train[::2,...]
#y_train = y_train[::2]


filename = data_type+'_sublist_train_mean.hdf5'
#filename = data_type+'_sublist_train_mean_FA.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist,hdf5_format)

filename = data_type+'_sublist_test.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_test,y_test) = ids_dataset.load_data(filelist,hdf5_format)

filename = data_type+'_sublist_val.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_val,y_val) = ids_dataset.load_data(filelist,hdf5_format)
#%%Select the scan to create RGB image
#input shape (N,CH,X,Z) 
z0 = int(np.floor(x_train.shape[3]/2.))
x0 = int(np.floor(x_train.shape[2]/2.))
y0 = int(np.floor(x_train.shape[1]/2.))
if scan_type == 'B':
    slice_vect = range(y0-1,y0+2)#last index is not taken
    x_train = x_train[:,slice_vect,:,:]
    x_test = x_test[:,slice_vect,:,:]
    x_val = x_val[:,slice_vect,:,:]
    x_mu = x_mu[:,slice_vect,:,:]
    #output shape (N,Z,X,CH)
    x_train = x_train.transpose((0,3,2,1))
    x_test = x_test.transpose((0,3,2,1))
    x_val = x_val.transpose((0,3,2,1))
    x_mu = x_mu.transpose((0,3,2,1))
elif scan_type == 'T':
    slice_vect = range(x0-1,x0+2)#last index is not taken
    x_train = x_train[:,:,slice_vect,:]
    x_test = x_test[:,:,slice_vect,:]
    x_val = x_val[:,:,slice_vect,:]
    x_mu = x_mu[:,:slice_vect,:]
    #output shape (N,Z,CH,X)
    x_train = x_train.transpose((0,3,1,2))
    x_test = x_test.transpose((0,3,1,2))
    x_val = x_val.transpose((0,3,1,2))
    x_mu = x_mu.transpose((0,3,1,2))
elif scan_type == 'C':
    slice_vect = range(z0-1,z0+2)#last index is not taken
    x_train = x_train[:,:,:,slice_vect]
    x_test = x_test[:,:,:,slice_vect]
    x_val = x_val[:,:,:,slice_vect]
    x_mu = x_mu[:,:,:,slice_vect]
    #output shape (N,CH,X,Z)
    
#%%subtract mean before training ans rescale in [-1 1]
x_train = (x_train - x_mu)/max_value
x_test = (x_test - x_mu)/max_value
x_val = (x_val - x_mu)/max_value

if not(y_train.dtype) == 'uint8':
    y_train=y_train.astype('uint8')
    y_test=y_test.astype('uint8')
    y_val=y_val.astype('uint8')
    print('converting label to uint8')


#%%
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#model_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#model_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396-05-0.95.hdf5'
#model_to_load = 'ids_Cscan0e01sgd1523024265-30-0.85.hdf5'
#model = load_model(model_to_load)

epochs_added = 20

dataset_id = ids.get_dataset_id(dataset_dir)
model_id = '0e01sgd1523024265'
net_id = '64_128_33sub' #512
#net_id = '32_64_128_333sub' #512x512
#net_id = '32_64_128_333' #512x512
#net_id = '32_32_32_333' #128x128
epoch_best = 30
val_best = 0.85
filepath='ids_{}scan{}-{:02d}-{:.2f}.hdf5'.format(scan_type,model_id,epoch_best,val_best)

callbacks_list = []
tensorboard = TensorBoard(log_dir='logs/'+model_id+net_id+dataset_id)
callbacks_list.append(tensorboard)
# checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=True, save_best_only=True,save_weights_only=False, mode='max',period=1)
callbacks_list.append(checkpoint)

model = load_model(filepath)

#batch_size_train,iter_train,rest_train = ids.get_optimum_batch(x_train)
batch_size_train = 32
start = time()
history = model.fit(x_train, y_train, batch_size=batch_size_train, epochs=epochs_added,validation_data=(x_test, y_test),callbacks=callbacks_list)
end = time()
took = end - start
print ('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
