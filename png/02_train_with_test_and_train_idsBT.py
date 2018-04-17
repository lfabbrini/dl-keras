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
#import re
from time import time
from support import ids_dataset
from support import ids

scan_type = 'B'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_train_and_test.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train_B,y_train_B) = ids_dataset.load_data(filelist)

filename = scan_type+'_sublist_val.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_test_B,y_test_B) = ids_dataset.load_data(filelist)

scan_type = 'T'
filename = scan_type+'_sublist_train_and_test.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train_T,y_train_T) = ids_dataset.load_data(filelist)

filename = scan_type+'_sublist_val.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_test_T,y_test_T) = ids_dataset.load_data(filelist)

x_train = np.concatenate((x_train_B,x_train_T),axis=0)
del x_train_B
del x_train_T
y_train = np.vstack((y_train_B,y_train_T))
del y_train_B
del y_train_T

x_test = np.concatenate((x_test_B,x_test_T),axis=0)
del x_test_B
del x_test_T
y_test = np.vstack((y_test_B,y_test_T))
del y_test_B
del y_test_T

x_mu = x_train.sum(axis=0)/x_train.shape[0]

#filename = scan_type+'_sublist_val.npz'
#filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
#(x_val,y_val) = ids_dataset.load_data(filelist)

#%%Debug
plt.imshow(x_train[0])
#plt.imshow(np.squeeze(x_train[0,:,:,1]),cmap='Greys')
plt.show()

plt.imshow(x_test[1])
plt.show()

#plt.imshow(x_val[2])
#plt.show()

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(x_mu[:,:,i]),cmap='Greys')
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()



#%%subtract mean before training ans rescale in [-1 1]
x_train = (x_train - x_mu)/255
x_test = (x_test - x_mu)/255
#x_val = (x_val - x_mu)/255

if not(y_train.dtype) == 'uint8':
    y_train=y_train.astype('uint8')
    y_test=y_test.astype('uint8')
#    y_val=y_val.astype('uint8')
    print('converting label to uint8')
#add color depth
if grayscale and x_train.ndim==3:
    x_train = x_train.reshape(x_train.shape+ (1,))
    x_test = x_test.reshape(x_test.shape+ (1,))
#    x_val = x_val.reshape(x_val.shape+ (1,))
    print('added new axes')
#%%
from keras import layers
from keras import models
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard

#%% Solver
#dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
#lr = 0.01
#optimizer = 'sgd'
#this_conf = {'lr':lr, 'optimizer':optimizer}
#lr_list = [0.01, 0.001,0.1, ]

padding='same'
#padding='valid'
lr_list = [0.01]
optimizer_list = ['sgd']

#net_id = '32_64_128_333nopad' #512x512
net_id = '32_32_32_333_trteBT' #128x128
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
    model.add(layers.Conv2D(32, (3, 3), padding=padding, activation='relu', input_shape=x_train.shape[1:]))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), padding=padding, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), padding=padding, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.summary()
    #%% Classifier
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
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
    with open('{}/summary{}.txt'.format(cwd,model_id+net_id),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))    
        
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='logs/'+model_id+net_id+dataset_id)
    
    batch_size_train,iter_train,rest_train = ids.get_optimum_batch(x_train)
    start = time()
    history = model.fit(x_train, y_train, batch_size=batch_size_train, epochs=30,validation_data=(x_test, y_test),callbacks=[tensorboard])
    end = time()
    took = end - start
    print ('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
    try:
        model.save('ids_{}scan{}.h5'.format(scan_type,model_id))
    except:
        pass
#    batch_size_val,iter_val,rest_val = ids.get_optimum_batch(x_val)
#    score = model.evaluate(x_val, y_val, batch_size=batch_size_val)
#    with open('{}/summary{}.txt'.format(cwd,model_id+net_id),'a') as fh:
#        fh.write('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
#        fh.write('score on val: {:f}',format(score))
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
