# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
from keras import layers
from keras import models


def VGG(input_shape,filter_size=[32,64,128],kernel_size=[3,3,3],padding='same',activation='relu',max_pool=[True,True,True],batch_norm=[True,True,True],dense_size=[512,512],conv_type='2D',kernel_regularizer=None):
    model = models.Sequential()
    id_str=''
  
     
        
    for i,zipped in enumerate(zip(filter_size,kernel_size,max_pool,batch_norm)):
        (f,k,mp,bn) = zipped
        if i==0:
            if conv_type == '2D':
                model.add(layers.Conv2D(f, (k, k), padding=padding, kernel_regularizer=kernel_regularizer, input_shape=input_shape))     
            elif conv_type == '2Dsep':
                model.add(layers.SeparableConv2D(f, (k, k), padding=padding, kernel_regularizer=kernel_regularizer, input_shape=input_shape))     
            elif conv_type == '3D':    
                model.add(layers.Conv3D(f, (k, k, k), padding=padding, kernel_regularizer=kernel_regularizer, input_shape=input_shape))     
        else:
            if conv_type == '2D':
                model.add(layers.Conv2D(f, (k, k), padding=padding , kernel_regularizer=kernel_regularizer))     
            elif conv_type == '2Dsep':
                model.add(layers.SeparableConv2D(f, (k, k), padding=padding , kernel_regularizer=kernel_regularizer))    
            elif conv_type == '3D':    
                model.add(layers.Conv3D(f, (k, k, k), padding=padding , kernel_regularizer=kernel_regularizer))     
            
        model.add(layers.Activation(activation))
        if mp:
            if conv_type == '3D':
                model.add(layers.MaxPooling3D((2, 2, 2)))
            else:
                model.add(layers.MaxPooling2D((2, 2)))
        if bn:    
            model.add(layers.BatchNormalization(axis=-1))
        
        id_str += '(f{}k{}m{}b{})'.format(f,k,int(mp),int(bn))
    
    model.add(layers.Flatten())

    for d in dense_size:
         model.add(layers.Dense(d))
         model.add(layers.Activation(activation))
         id_str += 'd{}'.format(d)
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    #add regularization
    if kernel_regularizer is not None:
        config = kernel_regularizer.get_config()
        if config['l2']:
            id_str += 'l2{:1.0e}'.format(config['l2'])
        if config['l1']:
            id_str += 'l1{:1.0e}'.format(config['l1'])    
    #add conv type
    id_str += '_'+conv_type
    return model,id_str