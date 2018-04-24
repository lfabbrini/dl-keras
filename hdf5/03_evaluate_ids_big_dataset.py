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
#sys.path.append('C:/Users/l.fabbrini/dl-keras/')
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
from support.evaluation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


#Debug
#hdf5_format = True
#(x_mu,y_mu) = ids_dataset.load_data(file_to_mean,hdf5_format)

#%%Load model
from keras.models import load_model
#modelweight_dir = 'C:/Users/l.fabbrini/spyder/hdf5/'
modelweight_dir = '/home/mmessina/dl-keras/hdf5/'
#modelweight_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#modelweight_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#modelweight_to_load = 'ids_Cscan0e001sgd1521723396.h5'
#modelweight_to_load = 'ids_Cscan0e001sgd1521723396-05-0.95.hdf5'
#modelweight_to_load = 'ids_Cscan0e01sgd1521802004-10-0.93.hdf5'
#modelweight_to_load = 'ids_VC0e01sgd1523870966_32_64_128_333batch-01-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523955541_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-08-0.93.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523953785_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-05-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523960441_(f8k3m0b1)(f8k3m0b1)(f8k3m1b1)d512d512_2Dsep-02-0.93.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523964723_(f8k3m0b1)(f8k3m0b1)(f8k3m1b1)(f8k3m1b1)d256d256_2Dsep-14-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523965402_(f32k3m0b1)(f16k3m0b1)(f16k3m1b1)d256d256_2Dsep-07-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523965767_(f8k3m0b1)(f8k3m0b1)(f8k3m1b1)d256_2Dsep-06-0.93.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523966380_(f8k3m0b1)(f8k3m0b1)(f8k3m1b1)d512d512_3D-03-0.93.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1523958898_(f16k3m1b1)(f32k3m1b1)(f64k3m1b1)d512d512l21e-03l11e-02_2Dsep-12-0.91.hdf5'

#modelweight_to_load = 'ids_VB0e005sgd1523955807_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-02-0.88.hdf5'
#modelweight_to_load = 'ids_VB0e005sgd1523956162_(f64k3m1b1)(f64k3m1b1)(f64k3m1b1)d512d512_2Dsep-05-0.87.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524141054_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-09-0.90.hdf5'
#modelweight_to_load = 'ids_VC0e005rmsprop1524141754_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-10-0.90.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1524143352_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-05-0.89.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524144397_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-10-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524146812_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-02-0.95.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524147741_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-13-0.90.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524148739_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d256d256_2Dsep-05-0.92.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524149419_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d256d256_2Dsep-09-0.91.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524150233_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-13-0.92.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1524215735_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-08-0.96.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524483766_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-03-0.95.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524484497_(f32k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_2Dsep-04-0.94.hdf5'
#modelweight_to_load = 'ids_VC0e005sgd1524493753_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-02-0.97.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1524494406_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-06-0.97.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1524495827_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-02-0.99.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1524494406_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-06-0.97.hdf5'
modelweight_to_load = 'ids_VT0e005sgd1523886008_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-08-0.91.hdf5'
conv_type='3D'#2D,2Dsep
#conv_type='2D'#2D,2Dsep
#conv_type='2Dsep'#2D,2Dsep
filename = os.path.join(modelweight_dir,modelweight_to_load)
model = load_model(filename)
print('model input shape: {}'.format(model.input_shape))
#%%
#scan_type = 'C'
scan_type = 'T'
#scan_type = 'B'

data_type = 'V'
#path_data = '/home/lfabbrini/data'
path_data = '/home/mmessina/data'
#path_data = '/media/sf_share/'
#path_data = 'C:/Users/l.fabbrini/share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
#model_dir = 'mdl_tr80val10te10fs1_tex1_0001_dz1_dz1' #downsampling_xyz = [1,1,1]
#model_dir = 'mdl_tr80val10te10fs1_tex1_0002_dz1_dz6' #downsampling_xyz = [1,1,6]
#model_dir = 'mdl_debug'
model_dir = 'mdl_tr80val10te10fs1_tex3_0001' #downsampling_xyz = [1,1,1] without T12 e T15




stacked_scan = 13
downsampling_xyz = [1,1,6]
filename = data_type+'_sublist_train_mean.hdf5'
#filename = data_type+'_sublist_train_mean_FA.hdf5'
file_to_mean = os.path.join(path_data,dataset_dir,model_dir,filename)


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

batch_size = 1 
filename_base = data_type+'_sublist_test'
filename_h5 = filename_base+'.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename_h5)
gen = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=False)

#batch_size = 1 
#filename_base = data_type+'_sublist_val'
#filename_h5 = filename_base+'.hdf5'
#file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename_h5)
#gen = ids.ScanDataGenerator(file_to_data=file_to_data,
#                                        file_to_mean=file_to_mean,
#                                        batch_size=batch_size,
#                                        scan_type=scan_type,
#                                        downsampling_xyz=downsampling_xyz,
#                                        stacked_scan=stacked_scan,
#                                        preprocessing_function_list = preprocessing_function_list,
#                                        shuffle=False)

#%% Extract central image indexes
#extract value of labels
filename = 'Label_Info.txt'
filelist = os.path.join(path_data,dataset_dir,filename)
table_label_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')

#read information of each x_train image
filename = filename_base+'_Info.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')


ID_Unique = table_info['ID_Unique']
AcqID = table_info['AcqID']
XYZ_shift_1 = table_info['XYZ_shift_1']
XYZ_shift_2 = table_info['XYZ_shift_2']
XYZ_shift_3 = table_info['XYZ_shift_3']
XYZ_flip_1 = table_info['XYZ_flip_1']
XYZ_flip_2 = table_info['XYZ_flip_2']


areCentral = np.absolute(XYZ_shift_1) + np.absolute(XYZ_shift_2) + np.absolute(XYZ_shift_3) + np.absolute(XYZ_flip_1) + np.absolute(XYZ_flip_2) == 0
pos = [i for i, xx in enumerate(areCentral) if xx]

#%%
#gen = test_generator

M = len(pos)
y_pred = np.zeros((M,1),dtype='uint8')
y_val = np.zeros((M,1),dtype='uint8')
p_pred = np.zeros((M,1),dtype='float32')
for i,p in enumerate(pos):
    (x,y) = gen[p]
    y_val[i] = y[0]
    p_pred[i] = model.predict(x)
    if p_pred[i] > 0.5:
        y_pred[i] = 1
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, y_pred)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]


# Plot non-normalized confusion matrix
class_names = ['T','FA']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
##Plot F1 (pos_label=0 indica che si vuole calcolare F1 per la classe 0)
#f1 = f1_score(y_test, y_pred, pos_label=0)
acc = accuracy_score(y_val, y_pred)
print ("{0:s}\t{1:2.3f}".format("acc",acc))
#print "{0:s}\t{1:2.3f}".format("f1",f1)
print(classification_report(y_val, y_pred, target_names=class_names))

central_tuple = [(ID_Unique[p],AcqID[p]) for p in pos]
areMiss = y_pred != y_val
miss_tuple = [(*central_tuple[i],p_pred[i][0]) for i,_ in enumerate(y_pred) if areMiss[i]]
#print(miss_tuple)


#%% Compute FAR/m2
pos_ind = np.asarray(pos)
pos_ind = np.expand_dims(pos_ind,axis=-1)
AcqID_miss = list(set(AcqID[pos_ind[areMiss]]) )

#Leggi Info AcqID MeterSquaredProcesed
filename = 'AcquisitionPerformanceInfo.txt'
filelist = os.path.join(path_data,dataset_dir,filename)
table_Acqinfo = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')
AcqID_acqinfo = table_Acqinfo['AcqID']
MeterSquaredProcesed_acqinfo = table_Acqinfo['MeterSquaredProcesed']


MeterSquaredProcesed_miss = [MeterSquaredProcesed_acqinfo[i] for i,(m,a) in enumerate(zip(MeterSquaredProcesed_acqinfo,AcqID_acqinfo)) for am in AcqID_miss if am == a ]
MeterSquaredProcesed_miss = np.asarray(MeterSquaredProcesed_miss)
MeterSquaredProcesed_miss = np.sum(MeterSquaredProcesed_miss)
Farm2 = cnf_matrix[1,0]/MeterSquaredProcesed_miss

print ('-'*40)
print ("{:s}\t".format(modelweight_to_load))
print ('-'*40)
print ("{:s}\t{:2.3f}".format("PD",cnf_matrix_norm[0,0]))
print ("{:s}\t{:1.5f}".format("FAR/m2",Farm2))
##Plot F1 (pos_label=0 indica che si vuole calcolare F1 per la classe 0)
f1 = f1_score(y_val, y_pred, pos_label=0)
acc = accuracy_score(y_val, y_pred)
print ("{:s}\t{:2.3f}".format("acc",acc))
print ("{:s}\t{:2.3f}".format("f1",f1))
#%% Error Analysis
file_to_mean = None #AVOID SUBTRACTING MEAN
batch_size = 1
filename_h5 = filename_base+'.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename_h5)
gen = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan,
                                        preprocessing_function_list = preprocessing_function_list,
                                        shuffle=False)

def show_image(x):
    
    N=int(np.ceil(np.sqrt(x.shape[-1])))
    eta = 3
    mu = np.mean(x)
    std = np.std(x)
    vmin = mu - eta*std
    vmax = mu + eta*std
    plt.figure(figsize=(10, 10))
    for i in range(N):
        for j in range(N):
            plt.subplot(N,N,1+j+i*N)
            if j+i*N < x.shape[-1]:
                plt.imshow(x[:,:,j+i*N],vmin=vmin,vmax=vmax)
#                plt.set_cmap('viridis')
                plt.set_cmap('gray')
#                plt.colorbar()
    plt.show()


def show_image_v2(x):
    CH = x.shape[-1]
    H = x.shape[-3]
    W = x.shape[-2]
    N=int(np.ceil(np.sqrt(CH)))
    margin = int(np.ceil(H*0.01))
    eta = 3
    mu = np.mean(x)
    std = np.std(x)
    vmin = mu - eta*std
    vmax = mu + eta*std
    results = np.zeros((N * H + (N-1) * margin, N * W + (N-1) * margin))
    for i in range(N):  # iterate over the rows of our results grid
        for j in range(N):  # iterate over the columns of our results grid
            ch = i*N + j
            if  ch >=  CH:
                continue
            # Put the result in the square `(i, j)` of the results grid
            h0 = i * H + i * margin
            h1 = h0 + H
            w0 = j * W + j * margin
            w1 = w0 + W
            results[h0: h1, w0: w1] = x[:,:,ch]
            
    fig = plt.figure()
#    plt.figure(figsize=(10, 10))
    plt.imshow(results,vmin=vmin,vmax=vmax)
    plt.set_cmap('gray')
    return fig
#err_pred = p_pred-y_val

areNotOk = y_pred != y_val
LabelToShow = table_label_info['T']
#LabelToShow = table_label_info['FA']
path_save = '/home/mmessina/'
for i,p in enumerate(pos):
    if areNotOk[i] and y_val[i] == LabelToShow:
        (x,y) = gen[p]
        
        p_pred = model.predict(x)        
        
        if conv_type=='3D':#for show_image
            x=x[...,0]

        #plt.figure()
        #title = scan_type+ID_Unique+str_augm
        #plt.title(title)
        #for i in range(x.shape[-1]): 
        #    plt.subplot(1,x.shape[-1],i+1)
        #    plt.imshow(x[:,:,i])
        #    plt.colorbar()
        #plt.show()
        #print(title, np.min(x), np.max(x))
        
        #sprintf('x%d_ch%d_y%d',shift_x,shift_ch,shift_y);
        str_shift = 'x{:d}_ch{:d}_y{:d}'.format(XYZ_shift_1[p],XYZ_shift_2[p],XYZ_shift_3[p])
        str_shift = re.sub(r'-','m',str_shift)
        
        #str_flip = sprintf('xf%dchf%d',xx-1,chch-1);
        str_flip = 'xf{:d}chf{:d}'.format(XYZ_flip_1[p],XYZ_flip_2[p])
        str_augm = str_shift + '_' + str_flip
        
        title = scan_type+ID_Unique[p]+str_augm
        print(title, np.min(x), np.max(x))
        print('Prob: {}'.format(p_pred))
#        show_image(x[0])
        fig = show_image_v2(x[0])
        filename = scan_type+ID_Unique[p]+'.png'
        file_to_save = os.path.join(path_save,filename)
        fig.savefig(file_to_save)
        

