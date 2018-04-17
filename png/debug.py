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
path_data = '/home/lfabbrini/data'
dataset_dir = 'NN_PNGrgbSScan_bal_m_wD_TgU_wUnkGT_P0e001__NAcq40_Tex4_201831211597/STC'
model_dir = 'mdl_tr80val10te10fs1_0001'
#model_dir = 'mdl_debug'
grayscale=False

filename = scan_type+'_sublist_train.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_train,y_train) = ids_dataset.load_data(filelist)
#downsampling by 2
#x_train = x_train[::2,...]
#y_train = y_train[::2]


filename = scan_type+'_sublist_train_mean.npz'
filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
(x_mu,y_mu) = ids_dataset.load_data(filelist)
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