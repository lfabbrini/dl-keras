#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:10:37 2018

@author: lfabbrini
"""


#%%
import sys
print (sys.version) #parentheses necessary in python 3.  
#print (sys.path)
sys.path.append('/home/lfabbrini/spyder/keras_wdir')
import os
#import h5py
from support import ids

import matplotlib.pyplot as plt
import numpy as np

data_type = 'V'
#path_data = '/home/lfabbrini/data'
path_data = '/media/sf_share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
model_dir = 'mdl_tr80val10te10fs1_tex1_0001'

scan_type = 'C'
stacked_scan = 3
downsampling_xyz = [1,1,6]
filename = data_type+'_sublist_train_mean.hdf5'
#filename = data_type+'_sublist_train_mean_FA.hdf5'
file_to_mean = os.path.join(path_data,dataset_dir,model_dir,filename)

#file_to_mean = None
batch_size=32
filename = data_type+'_sublist_val.hdf5'
file_to_data = os.path.join(path_data,dataset_dir,model_dir,filename)
val_generator = ids.ScanDataGenerator(file_to_data=file_to_data,
                                        file_to_mean=file_to_mean,
                                        batch_size=batch_size,
                                        scan_type=scan_type,
                                        downsampling_xyz=downsampling_xyz,
                                        stacked_scan=stacked_scan)
#%%
import h5py
f = h5py.File(file_to_data, 'r')
x_v, y_v = f['data'], f['label'] #x.shape is (N,H,W,CH)
idx = 0
x = x_v[idx * batch_size:(idx + 1) * batch_size]
y = y_v[idx * batch_size:(idx + 1) * batch_size]

#%%
mid_size_slice = int(np.floor(stacked_scan/2.))
ids.slice_volume(x,scan_type,mid_size_slice,downsampling_xyz)
#%%
(x,y) = val_generator.__getitem__(0)

areTarget = y == 0;
pos = [i for i, xx in enumerate(areTarget) if xx]
id_pos=1

filename = data_type+'_sublist_val_Info.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None)
print(table_info['ID_Unique'][pos[id_pos]])

col_ord = "RGB"
for i in range(3): 
    plt.subplot(1,3,i+1)
    plt.imshow(np.squeeze(x[pos[id_pos],:,:,i]))
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()
#%%
for i,(x,y) in enumerate(val_generator.__iter__()):
    break


#%%
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
import re
from time import time
from support import ids_dataset
from support import ids

scan_type = 'C'
#scan_type = 'T'
#scan_type = 'B'

data_type = 'V'
#path_data = '/home/lfabbrini/data'
path_data = '/media/sf_share/'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq40_Tex0_2018410112537/STC'
#model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
model_dir = 'mdl_tr80val10te10fs1_tex1_0001'
#model_dir = 'mdl_debug'
grayscale=False
hdf5_format=True
max_value = 1

filename = data_type+'_sublist_val.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_val,y_val) = ids_dataset.load_data(filelist,hdf5_format)

#%% Read sublist Info
x=x_val
y=y_val
filename = data_type+'_sublist_val_Info.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table_info = np.genfromtxt(filelist, names=True, delimiter=',', dtype=None)
filename = data_type+'_sublist_val.txt'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
table = np.genfromtxt(filelist, delimiter=' ', dtype=None)

areTarget = y == 0;
pos = [i for i, xx in enumerate(areTarget) if xx]
id_pos=1
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
    


filename = scan_type+'_'+table_info['ID_Unique'][pos[id_pos]]+str_augm+'.png'
dataset_dir_png = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC/'
filelist = os.path.join(path_data,dataset_dir_png,'img',filename)
img = ids.load_img(filelist,grayscale)
#img = ids.resize_img(img,(28,28))     
from keras.preprocessing.image import img_to_array #img_to_array
img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}

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

mid_size_slice = 1
downsampling_xyz = [1,1,6]
x=ids.slice_volume(x,scan_type,mid_size_slice,downsampling_xyz)


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

#%%
from scipy.misc import imresize
x_sat_h = 0.5
x_sat_l =-0.5
x_sat_h = (x_sat_h +10)/20
x_sat_l = (x_sat_l +10)/20
x = np.clip(x,x_sat_l,x_sat_h)
x = 255*(x -x_sat_l)/(x_sat_h-x_sat_l)

OSize = 28
def resize(x,Osize):
    N = x.shape[0]
    C = x.shape[-1]
    x_tr = np.zeros((N,OSize,OSize,C)) 
    for i in range(N):
        for c in range(C):
            x_tr[i,:,:,c] = imresize(x[i,:,:,c],(OSize,OSize))#should have 1 or 3 channel, convert to PIL image and go back
    return x_tr

x_r = resize(x,OSize)
#x_r = x_r.astype('uint8')
#x_r = img_to_array(x_r)
col_ord = "RGB"
for i in range(3): 
    plt.subplot(2,3,i+1)
    plt.imshow(np.squeeze(x_r[pos[id_pos],:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
    
    plt.subplot(2,3,3+i+1)
    plt.imshow(np.squeeze(img[:,:,i])/255)
    plt.colorbar()
    plt.title(col_ord[i])
plt.show()

max_value = 255
#%%
filename = data_type+'_sublist_train_mean.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist,hdf5_format)
#%%
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        
        
resample = _PIL_INTERPOLATION_METHODS[interpolation]
img = img.resize(width_height_tuple, resample)        

#%%
import sys
print(sys.path)
sys.path.append('/home/lfabbrini/spyder/keras_wdir')
from support import ids
import numpy as np

print(np.arange(11))

axis_size=11
mid_size_slice=2
downsampling = 4
print(ids.get_slice(axis_size,mid_size_slice,downsampling))

#%%
samples = np.arange(1,mid_size_slice+1)*downsampling
i0 = np.array([int(np.floor(axis_size/2.))])
left = i0-samples
right = i0+samples
slice_vect = np.concatenate((left[::-1] ,i0,right))
slice_vect>= 0
#%%
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

#scan_type = 'C'
#scan_type = 'T'
#scan_type = 'B'
scan_type = 'V'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_HDF5vol_0e75m_ext0e30_P0e001__NAcq3_Tex0_201845134725/STC'
model_dir = 'mdl_tr70val15te15fs1_tex3_0001'
#model_dir = 'mdl_debug'
grayscale=False
hdf5_format=True

filename = scan_type+'_sublist_train.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train,y_train) = ids_dataset.load_data(filelist,hdf5_format)

filename = scan_type+'_sublist_train_mean.hdf5'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist,hdf5_format)
#%%
im = 0
x = x_train[im]
y = y_train[im]
print(y)
x.shape
x.transpose().shape
x = x.transpose()#reverse
z0 = int(np.floor(x.shape[0]/2.));
x0 = int(np.floor(x.shape[1]/2.));
y0 = int(np.floor(x.shape[2]/2.));

plt.figure()
plt.imshow(np.squeeze(x[:,:,y0]))
plt.figure()
plt.imshow(np.squeeze(x[:,x0,:]))
plt.figure()
plt.imshow(np.squeeze(x[z0,:,:]))          

plt.figure()
plt.imshow(np.squeeze(x[:,:,y0-1:y0+2]*255))
plt.colorbar()
#%%
from support import ids_dataset
from support.evaluation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from support import ids
import numpy as np
import matplotlib.pyplot as plt
#scan_type = 'B'
scan_type = 'T'
#scan_type = 'C'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_val.txt'
(x_val_c,y_val_c),(anom_id,acq_id)=ids.get_central_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale)

from keras.preprocessing import image
import re
is_target_id_re = re.compile(r"T16")
#is_target_id_re = re.compile(r"FA21")
img = x_val_c[0]
for i,anom in enumerate(anom_id):
    is_target_id_res = is_target_id_re.search(anom)
    if is_target_id_res != None:
        img = x_val_c[i]
        print('image {} found!'.format(anom))
        break
img_c = np.squeeze(img[:,:,1])
#img_c = np.expand_dims(img_c,axis=2)
plt.figure()
plt.matshow(np.squeeze(img[:,:,1]),cmap='gray')
#plt.imshow(image.array_to_img(img_c))
plt.show()
#%%
conf=[]
lr = 0.01
optimizer = 'sgd'
this_conf = {'lr':lr, 'optimizer':optimizer}
conf.append(this_conf)
lr = 0.02
optimizer = 'rmsprop'
this_conf = {'lr':lr, 'optimizer':optimizer}
conf.append(this_conf)
print(conf)
#%%
import time
start = time.time()
lr_list = [0.1, 0.01, 0.001]
optimizer_list = ['sgd','rmsprop']
conf=[]
for lr in lr_list:
    for opt in optimizer_list:
        this_conf = {'lr':lr, 'optimizer':opt}
        conf.append(this_conf)

end = time.time()
took = end - start
print ('took {} time ({:d}h{:d}m{:s}s)'.format(took,int(took//3600),int(took//60),str(took-int(took))[2:]))
#%%
print("-"*10)
lr = str(0.01)
lr = lr.replace(".","e")
print(lr)

#%%
import re
from support import ids
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
num_acq_re = re.compile(r"NAcq\d+")
pfa_re = re.compile(r"P\de\d+")
dataset_id_re = re.compile(r"Tex\d+\_\d+")
data_re = re.compile(r"[a-zA-Z]+$")
fail_re = re.compile(r"aaandnadnadnan")

list_re = []
list_re.append(num_acq_re)
list_re.append(pfa_re)
list_re.append(dataset_id_re)
list_re.append(data_re)
list_re.append(data_re)
list_re.append(fail_re)
#for rr in list_re:
#    print (rr.search(dataset_dir).group())
    
print(ids.get_dataset_id(dataset_dir))
#%%
str_in = 'B_FA566_2017_07_07_003Swath1x0_ch0_y0.png'
str_in = 'B_T66_2017_07_07_003Swath1x0_ch0_y0.png'

import re
anomaly_id = re.compile(r"\_(T|FA)\d+\_")
acq_id = re.compile(r"_\d\d\d\d\_\d\d_\d\d[^x]*")
is_central = re.compile(r"x0\_ch0\_y0\.")
    
list_re = []
list_re.append(anomaly_id)
list_re.append(acq_id)
list_re.append(is_central)

for rr in list_re:
    print (rr.search(str_in).group())

str_out = anomaly_id.search(str_in).group()
print(str_out[1:])
#%%
from support import ids
scan_type = 'B'
#scan_type = 'T'
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_val.txt'
(x_val,y_val),(anom_id,acq_id)=ids.get_central_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale)
#%%
print("{:1.2f}".format(0.01))
#%%
import numpy as np
a=np.zeros((2,3,4),dtype='uint8')
b=np.zeros((5,3,4),dtype='uint8')
c=np.concatenate((a,b),axis=0)
print(c.shape)
x_mu = c.sum(axis=0)
print(x_mu,x_mu.shape)
a = np.zeros((3,1),dtype='uint8')
b = np.zeros((2,1),dtype='uint8')
c = np.vstack((a,b))
print(c.shape)


#%%
import numpy as np
a=int(1)
b=np.zeros(10,dtype='uint8')
b[0] = a

print(b)
#%%
#!/usr/bin/env python
# Prints when python packages were installed
from __future__ import print_function
from datetime import datetime
import os
import pip


if __name__ == "__main__":
    packages = []
    for package in pip.get_installed_distributions():
        package_name_version = str(package)
        try:
            module_dir = next(package._get_metadata('top_level.txt'))
            package_location = os.path.join(package.location, module_dir)
            os.stat(package_location)
        except (StopIteration, OSError):
            try:
                package_location = os.path.join(package.location, package.key)
                os.stat(package_location)
            except:
                package_location = package.location
        modification_time = os.path.getctime(package_location)
        modification_time = datetime.fromtimestamp(modification_time)
        packages.append([
            modification_time,
            package_name_version
        ])
    for modification_time, package_name_version in sorted(packages):
        print("{0} - {1}".format(modification_time,
                                 package_name_version))
#%%
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_labels = to_categorical(train_labels)