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

filename = scan_type+'_sublist_test.npz'#test to reply tensorboard performance
#filename = scan_type+'_sublist_val.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_val,y_val) = ids_dataset.load_data(filelist)

filename = scan_type+'_sublist_train_mean.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist)

x_val = (x_val - x_mu)/255
x_val_c = (x_val_c - x_mu)/255
#%%Load model
from keras.models import load_model
#model_to_load = 'ids_Bscan0e01sgd1521021533.h5'
#model_to_load = 'ids_Bscan0e01sgd1521101393.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396.h5'
#model_to_load = 'ids_Cscan0e001sgd1521723396-05-0.95.hdf5'
model_to_load = 'ids_Cscan0e01sgd1521802004-10-0.93.hdf5'
model = load_model(model_to_load)
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
M = len(y_val_c)
y_pred_c = np.zeros((M,1),dtype='uint8')
p_pred_c = model.predict(x_val_c,batch_size=1)
for i,p in enumerate(p_pred_c):
    if p > 0.5:
        y_pred_c[i] = 1
y_pred_c = y_pred_c.astype('uint8')
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val_c, y_pred_c)

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
acc = accuracy_score(y_val_c, y_pred_c)
print ("{0:s}\t{1:2.3f}".format("acc",acc))
#print "{0:s}\t{1:2.3f}".format("f1",f1)
print(classification_report(y_val_c, y_pred_c, target_names=class_names))
miss_tuple = [(anom,acq_id[i],p_pred_c[i]) for i,anom in enumerate(anom_id) if y_pred_c[i] != y_val_c[i]]
areNotOk = y_pred_c != y_val_c
for i,ok in enumerate(areNotOk):
    if ok==True:
        print((anom_id[i],acq_id[i],p_pred_c[i]))
#        plt.figure()
#        plt.imshow(np.squeeze(x_val_c[i,:,:,1]),cmap='Greys')
#        plt.colorbar()
#        plt.show()