"""IDS dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import h5py

def load_data(pathToFile,hdf5_format=False):
    """Loads the IDS dataset.

    # Arguments
        path: path where to cache the dataset locally

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train)`.
    """
    if hdf5_format:
        f = h5py.File(pathToFile, 'r')
        x, y = f['data'][:], f['label'][:]#slice with [:] to automatically convert h5py._hl.dataset.Dataset in numpy.ndarray
    else:  
        f = np.load(pathToFile)
        x, y = f['x'], f['y']
        f.close()
    return (x, y)


def load_data_all(scan_type,path_data,dataset_dir,model_dir,subtract_mean=True,hdf5_format=False):
    """Loads the IDS dataset as a whole.

    # Arguments
        scan_type: 'B','T','C','V'
        path_data: path where dataset_dir are located
        dataset_dir: folder name of the dataset
        model_dir: folder name of the model
        subtract_mean: wheter or not performing mean subtraction on data
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train),(x_test,y_test),(x_val,y_val)`.
    """
    
    if hdf5_format:
        ext = '.hdf5'
        max_value = 1
    else:
        ext = '.npz'
        max_value = 255
        
    filename = scan_type+'_sublist_train'+ext
    filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
    (x_train,y_train) = load_data(filelist,hdf5_format)
    
    if subtract_mean:
        filename = scan_type+'_sublist_train_mean'+ext
        filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
        (x_mu,y_mu) = load_data(filelist,hdf5_format)
    
    filename = scan_type+'_sublist_test'+ext
    filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
    (x_test,y_test) = load_data(filelist,hdf5_format)
    
    filename = scan_type+'_sublist_val'+ext
    filelist = os.path.join(path_data,dataset_dir,model_dir,filename)
    (x_val,y_val) = load_data(filelist,hdf5_format)
    
    ##%%subtract mean before training ans rescale in [-1 1]
    if subtract_mean:
        x_train = (x_train - x_mu)/max_value
        x_test = (x_test - x_mu)/max_value
        x_val = (x_val - x_mu)/max_value
    else:
        x_train = (x_train)/max_value
        x_test = (x_test)/max_value
        x_val = (x_val)/max_value
    
    if not(y_train.dtype) == 'uint8':
        y_train=y_train.astype('uint8')
        y_test=y_test.astype('uint8')
        y_val=y_val.astype('uint8')
        print('converting label to uint8')
        
    return (x_train,y_train),(x_test,y_test),(x_val,y_val)