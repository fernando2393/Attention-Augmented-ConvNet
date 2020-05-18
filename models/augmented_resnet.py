# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:15:46 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import sys
sys.path.append("..")

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
from layers.augmented_conv2d import augmented_conv2d

def augmented_resnet_layer(inputs,
                 F_out,
                 N_h,
                 depth_k,
                 depth_v,
                 relative_emb,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    x = inputs
    x = augmented_conv2d(x, F_out, kernel_size, strides, N_h, depth_k, depth_v, relative_emb)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x



def resnet50_att_augmented_v1(input_shape, depth, v, k, num_classes=10, n_last_layer_augmented=1, n_heads=1, relative_emb=True):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    F_out = 16
    num_res_blocks = int((depth - 2) / 6)

    curr_layer = 0
    max_depth = 3 * num_res_blocks
    
    inputs = Input(shape=input_shape)
    
    depth_k = k * F_out
    if curr_layer >= max_depth - n_last_layer_augmented:
        depth_v = 0
    else:
        depth_v = v * F_out
    x = augmented_resnet_layer(inputs, 
                           F_out, 
                           n_heads, 
                           depth_k, 
                           depth_v, 
                           relative_emb)
    curr_layer += 1

    
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
                
            if curr_layer >= max_depth - n_last_layer_augmented:
                depth_v = 0
            else:
                depth_v = v * F_out
            
            y = augmented_resnet_layer(x, 
                                       F_out, 
                                       n_heads, 
                                       depth_k, 
                                       depth_v, 
                                       relative_emb,
                                       strides = strides)
            curr_layer += 1
            
            if curr_layer >= max_depth - n_last_layer_augmented:
                depth_v = 0
            else:
                depth_v = v * F_out
            y = augmented_resnet_layer(x, 
                                       F_out, 
                                       n_heads, 
                                       depth_k, 
                                       depth_v, 
                                       relative_emb,
                                       activation = None)
            curr_layer += 1

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                if curr_layer >= max_depth - n_last_layer_augmented:
                    depth_v = 0
                else:
                    depth_v = v * F_out
                x = augmented_resnet_layer(x, 
                           F_out, 
                           n_heads, 
                           depth_k, 
                           depth_v, 
                           relative_emb,
                           kernel_size = 1,
                           strides = strides,
                           activation = None,
                           batch_normalization = False)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        F_out *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
    