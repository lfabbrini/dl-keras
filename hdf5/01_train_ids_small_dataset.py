#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)
sys.path.append('C:/Users/l.fabbrini/spyder/')
#python -m tensorboard.main --logdir="C:\Users\l.fabbrini\spyder\hdf5\logs"


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
#path_data = '/media/sf_share/'
path_data = 'c:/Users/l.fabbrini/share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
#model_dir = 'mdl_tr80val10te10fs1_tex1_0001_dz1_dz1' #downsampling_xyz = [1,1,1]
model_dir = 'mdl_tr80val10te10fs1_tex1_0002_dz1_dz6' #downsampling_xyz = [1,1,6]
#model_dir = 'mdl_debug'
grayscale=False
hdf5_format=True
max_value = 1


filename = data_type+'_sublist_train.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train,y_train) = ids_dataset.load_data(filelist,hdf5_format)


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

#%%Debug (show first target in x_train)
x=x_train
y=y_train
id_pos = 0

#read information of each x_train image
filename = data_type+'_sublist_train_Info.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')

#extract value of labels
filename = 'Label_Info.txt'
filelist = os.path.join(path_data,dataset_dir,filename)
table_label_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')

#find target position in y
areTarget = y == table_label_info['T']
pos = [i for i, xx in enumerate(areTarget) if xx]
im = pos[id_pos]
ID_Unique = table_info['ID_Unique'][im]

#input shape (N,CH,X,Z) 
z0 = int(np.floor(x.shape[3]/2.))
x0 = int(np.floor(x.shape[2]/2.))
y0 = int(np.floor(x.shape[1]/2.))

x = x[im]
y = y[im]

plt.figure()
plt.imshow(np.transpose(x[y0,:,:]))
plt.title('B_'+ID_Unique)
plt.show()

plt.figure()
plt.imshow(np.transpose(x[:,x0,:]))
plt.title('T_'+ID_Unique)
plt.show()

plt.figure()
plt.imshow(np.squeeze(x[:,:,z0]))  
plt.title('C_'+ID_Unique)
plt.show()   

#z0 = int(np.floor(x_mu.shape[3]/2.))
#x0 = int(np.floor(x_mu.shape[2]/2.))
#y0 = int(np.floor(x_mu.shape[1]/2.))
col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    if scan_type == 'B':
        plt.imshow(np.transpose(x_mu[0,y0-1+i,:,:]))
    elif scan_type == 'T':
        plt.imshow(np.transpose(x_mu[0,:,x0-1+i,:]))
    elif scan_type == 'C':
        plt.imshow(np.squeeze(x_mu[0,:,:,z0-1+i]))
    plt.colorbar()
    plt.title('mu_'+col_ord[i])
plt.show()
#%%Select the scan to create RGB image
#input shape (N,CH,X,Z) 
downsampling_xyz = np.array([1,1,1])
#downsampling_xyz = np.array([1,1,6])
mid_size_slice = 1
x_train = ids.slice_volume(x_train,scan_type,mid_size_slice,downsampling_xyz)
x_test = ids.slice_volume(x_test,scan_type,mid_size_slice,downsampling_xyz)
x_val = ids.slice_volume(x_val,scan_type,mid_size_slice,downsampling_xyz)
x_mu = ids.slice_volume(x_mu,scan_type,mid_size_slice,downsampling_xyz)

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
from keras import layers
from keras import models
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
#%% Solver
#dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
#lr = 0.01
#optimizer = 'sgd'
#this_conf = {'lr':lr, 'optimizer':optimizer}
#lr_list = [0.01, 0.001,0.1, ]

padding='same'
#padding='valid'
#lr_list = [0.1]
#lr_list = [0.01]
lr_list = [0.01]
optimizer_list = ['sgd']
epochs=30

#net_id = '64_128_33batch' #512
#net_id = '32_64_128_333sub' #512x512
net_id = '32_64_128_333' #512x512
#net_id = '32_64_128_333_batch' #512x512
#net_id = '32_32_32_333' #128x128
conf_list=[]
for lr in lr_list:
    for opt in optimizer_list:
        conf_list.append({'lr':lr, 'optimizer':opt})

for conf in conf_list:
    lr = conf['lr']
    optimizer = conf['optimizer']
    #optimizer = 'rmsprop'
    if optimizer == 'sgd':
        opti = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'rmsprop':
        opti = RMSprop(lr=lr)
        
        
    #%% CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding=padding, input_shape=x_train.shape[1:]))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
#    model.add(layers.BatchNormalization(axis=-1))
    
    model.add(layers.Conv2D(64, (3, 3), padding=padding))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
#    model.add(layers.BatchNormalization(axis=-1))
    
    model.add(layers.Conv2D(128, (3, 3), padding=padding))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
#    model.add(layers.BatchNormalization(axis=-1))
    
    model.add(layers.Flatten())
    model.summary()
    #%% Classifier
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
#    model.add(layers.BatchNormalization(axis=-1))
        
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
#    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()    
    
    #Create IDs
    model_id = "{:s}{:s}{}".format(str(lr).replace('.','e'),optimizer,int(time()))
    dataset_id = ids.get_dataset_id(dataset_dir)
    print("-"*30)
    print("{:s}".format(dataset_id))
    print("{:s}".format(model_id))
    print("-"*30)
    
    # Open the file
    cwd = os.getcwd() #directory where the script is called by terminal  
    with open('{}/summary{}.txt'.format(cwd,model_id+'_'+net_id),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))    
        
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    callbacks_list = []
    tensorboard = TensorBoard(log_dir='logs/'+model_id+'_'+net_id+dataset_id)
    callbacks_list.append(tensorboard)
    # checkpoint
    filepath='ids_{}scan{}'.format(scan_type,model_id) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=True, save_best_only=True,save_weights_only=False, mode='max',period=1)
    callbacks_list.append(checkpoint)
    
    #batch_size_train,iter_train,rest_train = ids.get_optimum_batch(x_train)
    batch_size_train = 32
    start = time()
    history = model.fit(x_train, y_train, batch_size=batch_size_train, epochs=epochs,validation_data=(x_test, y_test),callbacks=callbacks_list)
    end = time()
    took = end - start
    print ('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
    try:
        model.save('ids_{}scan{}.h5'.format(scan_type,model_id))
    except:
        pass
    batch_size_val,iter_val,rest_val = ids.get_optimum_batch(x_val)
    score = model.evaluate(x_val, y_val, batch_size=batch_size_val)
    with open('{}/summary{}.txt'.format(cwd,model_id+'_'+net_id),'a') as fh:
        fh.write('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
        fh.write('score on val: {:f}'.format(score[1]))
    #%%
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    #x_tt = np.append(x_train,x_test,axis=0)
    #y_tt = np.append(y_train,y_test,axis=0)
    #batch_size_tt,iter_tt,rest_tt = ids.get_optimum_batch(x_tt)
    #history = model.fit(x_tt, y_tt, batch_size=batch_size_tt, epochs=3,validation_data=(x_val, y_val),callbacks=[tensorboard])
    #%%  
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    if len(loss) > 1:
        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.show()
    
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']
#    epochs = range(len(loss))
#    if len(loss) > 1:
#        plt.figure()
#        plt.plot(epochs, loss, 'bo', label='Training loss')
#        plt.plot(epochs, val_loss, 'b', label='Validation loss')
#        plt.title('Training and validation loss')
#        plt.legend()
#        plt.show()
