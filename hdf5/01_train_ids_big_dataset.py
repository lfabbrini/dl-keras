#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)
#sys.path.append('C:/Users/l.fabbrini/spyder/')
sys.path.append('/home/mmessina/dl-keras/')
#python -m tensorboard.main --logdir="C:\Users\l.fabbrini\spyder\hdf5\logs"

import keras
print (keras.__version__)
#from keras.preprocessing import image #img_to_array
import matplotlib.pyplot as plt
import numpy as np

import os
import re
from time import time
from support import ids_dataset
from support import ids

scan_type = 'C'
#scan_type = 'T'
#scan_type = 'B'

data_type = 'V'
#path_data = '/home/lfabbrini/data'
path_data = '/home/mmessina/data'
#path_data = '/media/sf_share/'
#path_data = 'c:/Users/l.fabbrini/share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
model_dir = 'mdl_tr80val10te10fs1_tex1_0001_dz1_dz1' #downsampling_xyz = [1,1,1]
#model_dir = 'mdl_tr80val10te10fs1_tex1_0002_dz1_dz6' #downsampling_xyz = [1,1,6]
#model_dir = 'mdl_debug'




conv_type='3D'#2D,2Dsep
#conv_type='2D'#2D,2Dsep
#conv_type='2Dsep'#2D,2Dsep
stacked_scan = 9
downsampling_xyz = [1,1,6]
filename = data_type+'_sublist_train_mean.hdf5'
#filename = data_type+'_sublist_train_mean_FA.hdf5'
file_to_mean = os.path.join(path_data,dataset_dir,model_dir,filename)


#Debug
hdf5_format = True
(x_mu,y_mu) = ids_dataset.load_data(file_to_mean,hdf5_format)



#%%Preprocessing  To have same PNG performance (OBS: z have the double of sample than .png image in older dataset) 
from functools import partial
#clip data to x_sat,-x_sat
#input are in [0,1] representing value in [-10,10]
#map x_sat from [-10,10] to [0,1]
x_sat_h = 0.5
x_sat_l =-0.5
x_sat_h = (x_sat_h +10)/20
x_sat_l = (x_sat_l +10)/20


clip_ = partial(ids.clip,x_min=x_sat_l,x_max=x_sat_h)
linearmap_ = partial(ids.linearmap,x_min=x_sat_l,x_max=x_sat_h,y_min=0,y_max=255)
resize_ = partial(ids.resize,hsize=28,wsize=28)
divide = lambda x,den=255: x/den
divide_ = partial(divide,den=255)




preprocessing_function_list = []
preprocessing_function_list.append(clip_)
preprocessing_function_list.append(linearmap_)
preprocessing_function_list.append(resize_)
preprocessing_function_list.append(divide_)
#%% 
preprocessing_function_list = None
if conv_type=='3D':
    if preprocessing_function_list is None:
        preprocessing_function_list = []
#    add_channel = lambda x: x[...,np.newaxis]
    add_channel_ = partial(np.expand_dims,axis=-1)
    preprocessing_function_list.append(add_channel_)
batch_size=32
filename = data_type+'_sublist_train.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename)
train_generator = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=True)

batch_size = None #selected automatically
filename = data_type+'_sublist_test.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename)
test_generator = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=False)

batch_size = None #selected automatically
filename = data_type+'_sublist_val.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename)
val_generator = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=False)

#%%Debug (show first target in x_train)
id_pos = 0
filename = data_type+'_sublist_train.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename)
train_generator_no_shuffle = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=False)
(x,y) = train_generator_no_shuffle[0]



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
XYZ_shift_1 = table_info['XYZ_shift_1'][im]
XYZ_shift_2 = table_info['XYZ_shift_2'][im]
XYZ_shift_3 = table_info['XYZ_shift_3'][im]
XYZ_flip_1 = table_info['XYZ_flip_1'][im]
XYZ_flip_2 = table_info['XYZ_flip_2'][im]

#sprintf('x%d_ch%d_y%d',shift_x,shift_ch,shift_y);
str_shift = 'x{:d}_ch{:d}_y{:d}'.format(XYZ_shift_1,XYZ_shift_2,XYZ_shift_3)
str_shift = re.sub(r'-','m',str_shift)

#str_flip = sprintf('xf%dchf%d',xx-1,chch-1);
str_flip = 'xf{:d}chf{:d}'.format(XYZ_flip_1,XYZ_flip_2)
str_augm = str_shift + '_' + str_flip

#B input shape (N,Z,X,CH)
#T input shape (N,Z,CH,X)
#C input shape (N,CH,X,Z) 
x = x[im]
y = y[im]

if conv_type=='3D':
    x=x[...,0]
    


c0 = int(np.floor(x.shape[-1]/2.))


def show_image(x):
    plt.figure()
    N=int(np.ceil(np.sqrt(x.shape[-1])))
    for i in range(N):
        for j in range(N):
            plt.subplot(N,N,1+j+i*N)
            if j+i*N < x.shape[-1]:
                plt.imshow(x[:,:,j+i*N])
                plt.set_cmap('viridis')
#                plt.colorbar()
    plt.show()
#plt.figure()
#title = scan_type+ID_Unique+str_augm
#plt.title(title)
#for i in range(x.shape[-1]): 
#    plt.subplot(1,x.shape[-1],i+1)
#    plt.imshow(x[:,:,i])
#    plt.colorbar()
#plt.show()
#print(title, np.min(x), np.max(x))

title = scan_type+ID_Unique+str_augm
print(title, np.min(x), np.max(x))
show_image(x)

mid_size_slice = int(np.floor(stacked_scan/2.))
x_mu_sl = ids.slice_volume(x_mu,scan_type,mid_size_slice,downsampling_xyz)
if preprocessing_function_list:
            for i,f in enumerate(preprocessing_function_list):
                x_mu_sl = f(x_mu_sl)

x_mu_sl = x_mu_sl[0]

if conv_type=='3D':
    x_mu_sl=x_mu_sl[...,0]
#plt.figure()
#title = 'mu'
#plt.title(title)
#for i in range(x.shape[-1]):
#    plt.subplot(1,x.shape[-1],i+1)
#    plt.imshow(x_mu_sl[:,:,i])
#    plt.colorbar()
#plt.show()
#print(title, np.min(x_mu_sl), np.max(x_mu_sl))

title = 'mu'
print(title, np.min(x_mu_sl), np.max(x_mu_sl))
show_image(x_mu_sl)
#%%
from support import netzoo
from keras import regularizers
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
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
lr_list = [0.005]
optimizer_list = ['sgd']
epochs=30
kernel_regularizer = regularizers.l2(0.001)
kernel_regularizer = None

#net_id = '64_64_64_333' #128x128
#net_id = '64_128_33batch' #512
#net_id = '32_64_128_333sub' #512x512
#net_id = '32_64_128_333' #512x512
#net_id = '32_64_128_333batch' #512x512
#net_id = '32_32_32_333' #128x128

#filter_size=[16,16,16]
#max_pool=[False,True,True]

filter_size=[8,8,8]
max_pool=[False,False,True]

kernel_size=[3,3,3,3]
batch_norm=[True,True,True,True]
dense_size=[512,512]

input_shape=train_generator.shape()[1:]


conf_list=[]
for lr in lr_list:
    for opt in optimizer_list:
        conf_list.append({'lr':lr, 'optimizer':opt})

for conf in conf_list:
    lr = conf['lr']
    optimizer = conf['optimizer']
    #optimizer = 'rmsprop'
    if optimizer == 'sgd':
        opti = SGD(lr=lr, decay=1e-6, momentum=0.9)
    elif optimizer == 'sgdn':
        opti = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'rmsprop':
        opti = RMSprop(lr=lr/100)
        
        
    #%% CNN
    model,net_id = netzoo.VGG(input_shape,kernel_size=kernel_size,filter_size=filter_size,max_pool=max_pool,dense_size=dense_size,conv_type=conv_type,batch_norm=batch_norm,kernel_regularizer=kernel_regularizer)
    model.summary()    
    key=input('Continue [Y/n]')
    if key == 'n':
        break;
    
    #Create IDs
    model_id = "{:s}{:s}{}".format(str(lr).replace('.','e'),optimizer,int(time()))
    dataset_id = ids.get_dataset_id(dataset_dir)
    print("-"*30)
    print("{:s}".format(dataset_id))
    print("{:s}".format(model_id))
    print("-"*30)
    
    final_id = data_type+scan_type+model_id+'_'+net_id
    #Write Summary--------------------------------------------
    # Open the file
    cwd = os.getcwd() #directory where the script is called by terminal  
    with open('{}/summary{}.txt'.format(cwd,final_id),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))    
        
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    #Callback-------------------------------------------------
    callbacks_list = []
     # TensorBoard
    tensorboard = TensorBoard(log_dir='logs/'+dataset_id+'_'+final_id,
#                              histogram_freq = 1,
#                              write_grads = True,
#                              batch_size = 32,
                              write_images = True)
    callbacks_list.append(tensorboard)
    # ModelCheckpoint
    filepath='ids_{}'.format(final_id) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=True, save_best_only=True,save_weights_only=False, mode='max',period=1)
    callbacks_list.append(checkpoint)
    # EarlyStopping
    earlystopping = EarlyStopping(monitor='acc',patience=5)
    callbacks_list.append(earlystopping)
    
    #-----------
    #input("Press Enter to continue ...")
    #-----------
    start = time()
    history = model.fit_generator(train_generator,epochs=epochs,validation_data=test_generator,callbacks=callbacks_list)
    end = time()
    took = end - start
    print ('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
    try:
        model.save('ids_{}scan{}.h5'.format(scan_type,model_id))
    except:
        pass
    
#    score = model.evaluate_generator(test_generator)
#    with open('{}/summary{}.txt'.format(cwd,model_id+'_'+net_id),'a') as fh:
#        fh.write('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
#        fh.write('score on val: {:f}'.format(score[1]))
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
