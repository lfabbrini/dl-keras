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

from support.ids import create_data_from_filelist
#from support import ids_dataset

# The directory where we will
# store our smaller dataset
#scan_type = 'B'
#scan_type = 'T'
scan_type = 'C'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_train.txt'
create_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale,compute_mean=True)

filename = scan_type+'_sublist_train_and_test.txt'
create_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale,compute_mean=True)

filename = scan_type+'_sublist_test.txt'
create_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale,compute_mean=False)

filename = scan_type+'_sublist_val.txt'
create_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale,compute_mean=False)

