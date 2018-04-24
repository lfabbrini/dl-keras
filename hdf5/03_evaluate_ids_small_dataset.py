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
sys.path.append('C:/Users/l.fabbrini/dl-keras/')
#sys.path.append('/home/mmessina/dl-keras/')
#python -m tensorboard.main --logdir="C:\Users\l.fabbrini\spyder\hdf5\logs"

import keras
print (keras.__version__)

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #do not shwo issue with pre-build binaries
#import re
from time import time
from support import ids_dataset
from support.evaluation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
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

filename_base = data_type+'_sublist_test'
filename_h5 = filename_base+'.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename_h5)
(x_val,y_val) = ids_dataset.load_data(filelist,hdf5_format)

filename = scan_type+'_sublist_train_mean.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist,hdf5_format)

x_val = (x_val - x_mu)/max_value
#%%Load model
from keras.models import load_model
modelweight_dir = 'C:/Users/l.fabbrini/spyder/hdf5/'
#modelweight_dir = 'C:/Users/l.fabbrini/spyder/png/'
#modelweight_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#modelweight_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#modelweight_to_load = 'ids_Cscan0e001sgd1521723396.h5'
#modelweight_to_load = 'ids_Cscan0e001sgd1521723396-05-0.95.hdf5'
#modelweight_to_load = 'ids_Cscan0e01sgd1521802004-10-0.93.hdf5'
modelweight_to_load = 'ids_VC0e01sgd1523870966_32_64_128_333batch-01-0.92.hdf5'
#modelweight_to_load = 'ids_Bscan0e01sgd1523648221-02-0.93.hdf5'
#modelweight_to_load = 'ids_Bscan0e01sgd1523635523-02-0.93.hdf5'
#modelweight_to_load = 'ids_VT0e005sgd1523886008_(f16k3m0b1)(f16k3m1b1)(f16k3m1b1)d512d512_3D-08-0.91.hdf5'

filename = os.path.join(modelweight_dir,modelweight_to_load)
model = load_model(filename)

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

#%% Non Central
#score = model.evaluate(x_val, y_val, batch_size=1)
M = len(y_val)
y_pred = np.zeros((M,1),dtype='uint8')
p_pred = model.predict(x_val,batch_size=1)
for i,p in enumerate(p_pred):
    if p > 0.5:
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


#%% Central
#score = model.evaluate(x_val, y_val, batch_size=1)
y_val = y_val_c
x_val = x_val_c
M = len(y_val)
y_pred = np.zeros((M,1),dtype='uint8')
p_pred = model.predict(x_val,batch_size=1)
for i,p in enumerate(p_pred):
    if p > 0.5:
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
areMiss = y_pred != y_val
central_tuple = (ID_Unique,AcqID)
miss_tuple = [(*central_tuple[i],p_pred[i][0]) for i,_ in enumerate(y_pred) if areMiss[i]]
#for i,ok in enumerate(areMiss):
#    if ok==True:
#        print((anom_id[i],acq_id[i],p_pred_c[i]))
##        plt.figure()
##        plt.imshow(np.squeeze(x_val_c[i,:,:,1]),cmap='Greys')
##        plt.colorbar()
##        plt.show()
        
        
#%% Compute FAR/m2
pos_ind = np.asarray([i for i,_ in enumerate(y_pred) if areMiss[i]])
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