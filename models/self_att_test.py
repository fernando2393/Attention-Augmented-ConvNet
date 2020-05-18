# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:51:01 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

from layers.augmented_layer import augmented_conv2d
from layers.self_attention import SelfAttention2D
#from layers.augmented_conv2d import augmented_conv2d
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Dense, Conv2D, Input
from tensorflow.keras.models import Model
from layers.augmented_layer import AttentionAugmentation2D


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
    x = augmented_conv2d(x, F_out, kernel_size=(3, 3), strides=(1, 1),
                     depth_k=0.2, depth_v=0.2, num_heads=2, relative_encodings=True)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x


def aug_con_2d(input_shape, num_classes):
    
    
    ip = Input(shape=input_shape)
    #sa = SelfAttention2D(2, 2, 2, True)
    #x = sa.forward_pass(ip)
    x = augmented_conv2d(ip, F_out=20, kernel_size=(3, 3),
                          k=0.2, v=0.2,  # k and v have to be floating point (if is 1, convert to float). Always convert to float to be sure
                          num_heads=4, relative_encodings=True)
    #x = augmented_conv2d(ip, 20, 3, 1, 4, depth_k=4, depth_v=4, relative=True)
    x = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax')(x)
    model = Model(ip, outputs)
    return model
    '''
    inputs = Input(shape=input_shape)
    F_out = 10
    n_heads = 2
    depth_k = 2
    
    depth_v = 2
    relative_emb = True
    # x = augmented_resnet_layer(inputs, 
    #                         F_out, 
    #                         n_heads, 
    #                         depth_k, 
    #                         depth_v, 
    #                         relative_emb,
    #                         strides = 1,
    #                         activation=None,
    #                         batch_normalization=False)
    x = AttentionAugmentation2D(depth_k, depth_v, n_heads)(inputs)
    
    #outputs = Dense(num_classes,
    #            activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=x)
    return model
    '''
    
    