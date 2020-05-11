# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:58:53 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

from tensorflow.keras.layers import Conv2D
from self_attention import SelfAttention2D
import tensorflow as tf


def augmented_conv2d(X, F_out, k, N_h, depth_k, depth_v, relative):
    """
    Augment conv2d by using self-attention features. It is possible that 
    all the features come from conv2d or from self-attention. This will 
    depend in the value of F_out and depth_v.

    Parameters
    ----------
    X : image tensor/s shape=[n_samples_batch, H, W, 3] 
    
    F_out : int
        Output depth.
    k : int
        kernel size.
    depth_k : int
        keys depth (all attention heads together). Same value for querys depth..
    depth_v : int
        values depth (all attention heads together).
    N_h : int
         number of heads.
    relative : boolean
        use 2d relative embedding or not.

    Returns
    -------
    tensor shape=[n_samples_batch, F_out]
        

    """
    n_conv_features = F_out - depth_v
    
    if n_conv_features == 0:  
        self_attent_out = SelfAttention2D(N_h, depth_k, depth_v, relative).forward_pass(X)
        return self_attent_out
        
    elif depth_v == 0:
        conv2d_out = Conv2D(filters=n_conv_features, kernel_size=k, padding='same')(X)
        return conv2d_out
        
    else: 
        conv2d_out = Conv2D(filters=n_conv_features, kernel_size=k, padding='same')(X)
        self_attent_out = SelfAttention2D(N_h, depth_k, depth_v, relative).forward_pass(X)
        augmented_conv2d_out = tf.concat([conv2d_out, self_attent_out], axis=3)
        return augmented_conv2d_out 

"""
import sys 

sys.path.append("..")
from cifar10_dataset.data_loader import get_train_val_test_datasets

x_train, y_train, x_val, y_val, x_test, y_test  = get_train_val_test_datasets(5000)


a = tf.convert_to_tensor(x_train[:5])

augmented_conv2d_out = augmented_conv2d(a, F_out=160, k=3, N_h=8, depth_k=160, depth_v=160, relative=True)
"""
