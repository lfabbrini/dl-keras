# -*- coding: utf-8 -*-

#%%
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)
sys.path.append('C:/Users/l.fabbrini/spyder/')

import keras
print (keras.__version__)
from keras.preprocessing import image #img_to_array
import matplotlib.pyplot as plt
import numpy as np

import os
import re
from time import time
from support import ids_dataset
from support import ids


#%% Read hdf5 Volume
scan_type = 'C'
#scan_type = 'T'
#scan_type = 'B'

data_type = 'V'
#path_data = '/home/lfabbrini/data'
#path_data = '/media/sf_share/'
path_data = 'c:/Users/l.fabbrini/share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
#model_dir = 'mdl_tr80val10te10fs1_tex1_0001' #downsampling_xyz = [1,1,1]
model_dir = 'mdl_tr80val10te10fs1_tex1_0002' #downsampling_xyz = [1,1,6]
#model_dir = 'mdl_debug'
grayscale=False
hdf5_format=True
max_value = 1

filename = data_type+'_sublist_val.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_val,y_val) = ids_dataset.load_data(filelist,hdf5_format)

#%%Extract ID and augmentation info of the ids_pos-th target
id_pos=1
x=x_val
y=y_val
filename = data_type+'_sublist_val_Info.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None,encoding='UTF-8')
filename = data_type+'_sublist_val.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table = np.genfromtxt(filelist, delimiter=' ', dtype=None,encoding='UTF-8')

areTarget = y == 0;
pos = [i for i, xx in enumerate(areTarget) if xx]

#XYZ_shift_1,XYZ_shift_2,XYZ_shift_3,XYZ_flip_1,XYZ_flip_2
XYZ_shift_1 = table_info['XYZ_shift_1'][pos[id_pos]]
XYZ_shift_2 = table_info['XYZ_shift_2'][pos[id_pos]]
XYZ_shift_3 = table_info['XYZ_shift_3'][pos[id_pos]]
XYZ_flip_1 = table_info['XYZ_flip_1'][pos[id_pos]]
XYZ_flip_2 = table_info['XYZ_flip_2'][pos[id_pos]]

#sprintf('x%d_ch%d_y%d',shift_x,shift_ch,shift_y);
str_shift = 'x{:d}_ch{:d}_y{:d}'.format(XYZ_shift_1,XYZ_shift_2,XYZ_shift_3)
str_shift = re.sub(r'-','m',str_shift)

#str_flip = sprintf('xf%dchf%d',xx-1,chch-1);
#str_flip = 'xf{:d}chf{:d}'.format(XYZ_flip_1,XYZ_flip_2)
#str_augm = str_shift + '_' + str_flip

str_flip = '';
if XYZ_flip_1 + XYZ_flip_2==1:
    str_flip = str_flip + 'flr'
if XYZ_flip_1 + XYZ_flip_2==2:
    str_flip = str_flip + 'f2'
str_augm = str_shift + str_flip    

#%%Read Corrisponding PNG
filename = scan_type+'_'+table_info['ID_Unique'][pos[id_pos]]+str_augm+'.png'
dataset_dir_png = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC/'
filelist = os.path.join(path_data,dataset_dir_png,'img',filename)
img = ids.load_img(filelist,grayscale)
#img = ids.resize_img(img,(28,28))     
from keras.preprocessing.image import img_to_array #img_to_array
img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}


#%%Show images
#------------------------------------
#PNG
#------------------------------------
print(filename)
#plt.figure()
#plt.imshow(np.squeeze(img[:,:,1]))
#plt.show()

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(img[:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(img), np.max(img))
print( np.min(img/255), np.max(img/255))
#------------------------------------
#HDF5
#------------------------------------
print(table[pos[id_pos]][0])
#input shape (N,CH,X,Z) 
downsampling_xyz = np.array([1,1,1])
#downsampling_xyz = np.array([1,1,6])
mid_size_slice = 1
Z = x.shape[3]
X = x.shape[2]
Y = x.shape[1]
z0 = int(np.floor(x.shape[3]/2.))
x0 = int(np.floor(x.shape[2]/2.))
y0 = int(np.floor(x.shape[1]/2.))
if scan_type == 'B':
    slice_vect = ids.get_slice(Y,mid_size_slice,downsampling_xyz[1])#last index is not taken
    x = x[:,slice_vect,:,:]
    #output shape (N,Z,X,CH)
    x = x.transpose((0,3,2,1))
elif scan_type == 'T':
    slice_vect = ids.get_slice(X,mid_size_slice,downsampling_xyz[0])#last index is not taken
    x = x[:,:,slice_vect,:]
    #output shape (N,Z,CH,X)
    x = x.transpose((0,3,1,2))
elif scan_type == 'C':
    slice_vect = ids.get_slice(Z,mid_size_slice,downsampling_xyz[2])#last index is not taken
    x = x[:,:,:,slice_vect]
    #output shape (N,CH,X,Z)


#plt.figure()
#plt.imshow(np.squeeze(x[pos[id_pos],:,:,1]))
#plt.show()


col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(x[pos[id_pos],:,:,i]))
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(x[pos[id_pos]]), np.max(x[pos[id_pos]]))
#%%Preprocessing   
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

preprocessing_function_list = []
preprocessing_function_list.append(clip_)
preprocessing_function_list.append(linearmap_)
preprocessing_function_list.append(resize_)


#%% Resample

#clip data to x_sat,-x_sat
#input are in [0,1] representing value in [-10,10]
#map x_sat from [-10,10] to [0,1]
img_h5 = x[pos[id_pos],...]
img_h5 = img_h5[np.newaxis,...]
if preprocessing_function_list:
    for i,f in enumerate(preprocessing_function_list):
        img_h5 = f(img_h5)
        print( np.min(img_h5), np.max(img_h5))
print( np.min(img_h5)/255, np.max(img_h5)/255)        
#%%
#PNG

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(img[:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(img), np.max(img))

#HDF5

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(img_h5[0,:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(img_h5), np.max(img_h5))


#%%
from scipy.misc import imresize #imresize
from keras import backend as K #img_to_array
col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(img_h5[0,:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(img_h5), np.max(img_h5))

x=img_h5
hsize,wsize = 28,28
xmin, xmax = np.min(x), np.max(x)
N = x.shape[0]
C = x.shape[-1]
x_r = np.zeros((N,hsize,wsize,C),dtype=K.floatx()) 
for i in range(N):
    for c in range(C):
        x_r[i,:,:,c] = imresize(x[i,:,:,c],(hsize,wsize))#should have 1 or 3 channel, convert to PIL image and go back
        
#imresize map [xmin,xmax] in [0,255] before resizing, so we have to put the limit back
x_r = ids.linearmap(x_r,0,255,xmin,xmax)

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(x_r[0,:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
print( np.min(x_r), np.max(x_r))