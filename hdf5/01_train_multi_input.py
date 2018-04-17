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
from keras.preprocessing import image #img_to_array
import matplotlib.pyplot as plt
import numpy as np

import os
#import re
from time import time
from support import ids_dataset
from support import ids

#scan_type = 'C'
#scan_type = 'T'

input

#%% Input 1

path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

scan_type = 'B'
(x_train_B,y_train_B),(x_test_B,y_test_B),(x_val_T,y_val_B) = ids_dataset.load_data_all(scan_type,path_data,dataset_dir,model_dir)
    
scan_type = 'T'
(x_train_T,y_train_T),(x_test_T,y_test_T),(x_val_T,y_val_T) = ids_dataset.load_data_all(scan_type,path_data,dataset_dir,model_dir)
    


#%%
from keras import layers
from keras import models
from keras import Input
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
#lr_list = [0.01]
lr_list = [0.01]
optimizer_list = ['sgd']
epochs=30

net_id = '32_64_128_333' #512x512
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
        
    #%%Functional CNN  
 
    #%%Sequential CNN
#    model = models.Sequential()
#    model.add(layers.Conv2D(32, (3, 3), padding=padding, input_shape=x_train.shape[1:]))
#    model.add(layers.Activation('relu'))
##    model.add(layers.BatchNormalization(axis=-1))
#    model.add(layers.MaxPooling2D((2, 2)))
#    
#    model.add(layers.Conv2D(64, (3, 3), padding=padding))
#    model.add(layers.Activation('relu'))
##    model.add(layers.BatchNormalization(axis=-1))
#    model.add(layers.MaxPooling2D((2, 2)))
#    
#    model.add(layers.Conv2D(128, (3, 3), padding=padding))
#    model.add(layers.Activation('relu'))
##    model.add(layers.BatchNormalization(axis=-1))
#    model.add(layers.MaxPooling2D((2, 2)))
#    
#    model.add(layers.Flatten())
#    model.summary()
#    #%% Classifier
#    model.add(layers.Dense(512))
#    model.add(layers.Activation('relu'))
##    model.add(layers.BatchNormalization(axis=-1))
#    model.add(layers.Dense(512))
#    model.add(layers.Activation('relu'))
##    model.add(layers.BatchNormalization(axis=-1))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.summary()    
    
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
    
    callbacks_list = []
    tensorboard = TensorBoard(log_dir='logs/'+model_id+net_id+dataset_id)
    callbacks_list.append(tensorboard)
    # checkpoint
    filepath='ids_{}scan{}'.format(scan_type,model_id) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,save_weights_only=False, mode='max',period=epochs//3)
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
    with open('{}/summary{}.txt'.format(cwd,model_id+net_id),'a') as fh:
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
