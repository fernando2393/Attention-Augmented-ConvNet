# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:39:16 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Flatten, MaxPool2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from layers.test_layer import augmented_conv2d
import numpy as np


def resnet_layer(inputs,
             num_filters=16,
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
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs

    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x

def resnet34_att_augmented(input_shape, num_classes, k, v, n_heads, relative_encoding=True):
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

    num_filters = 64
    inputs = Input(shape=input_shape)
    
    x = resnet_layer(inputs=inputs, 
                     num_filters = num_filters,
                     kernel_size=(7,7))
    
    x = MaxPool2D(pool_size=(3,3),
                  strides=2,
                  padding="same")(x)
    # Instantiate the stack of residual units
    num_res_blocks = [3, 4, 6, 3]
    for stack in range(4):
        for res_block in range(num_res_blocks[stack]):
            strides = (1, 1)
            # if stack > 0 and res_block == 0:  # first layer but not first stack
            #     strides = (2, 2)  # downsample
            if stack > 0:
                y = augmented_conv2d(x, F_out=num_filters, kernel_size=(3, 3),
                          k=float(k), v=float(v),
                          num_heads=n_heads, relative_encodings=relative_encoding,
                          strides=strides)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = augmented_conv2d(y, F_out=num_filters, kernel_size=(3, 3),
                          k=float(k), v=float(v),
                          num_heads=n_heads, relative_encodings=relative_encoding)
                y = BatchNormalization()(y)
            else:
                y = resnet_layer(inputs=x,
                                 kernel_size=(3,3),
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 kernel_size=(3,3),
                                 num_filters=num_filters,
                                 activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=(1,1),
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=True)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    #x = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model