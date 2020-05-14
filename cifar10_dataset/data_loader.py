# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:52:00 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import sys 

sys.path.append("..")

import numpy as np
from tensorflow.keras.datasets import cifar10
import tensorflow.keras


np.random.seed(123)

def preprocess_features(x_train, x_test, substract_pixel_mean=True):
    """
    Perform normalization (all values between 0 and 1) and mean normalization
    if substract_pixel_mean=True.

    Parameters
    ----------
    x_train : numpy array
    x_test :  numpy array
    substract_pixel_mean : boolean

    Returns
    -------
    x_train : numpy array
    x_test : numpy array

    """

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if substract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        return x_train, x_test, x_train_mean

    return x_train, x_test


def get_train_val_test_datasets(n_val_samples, categorical_targets=True, verbose=False): 
    """
    CIFAR-10 ready for training, validating and testing the model/s. 
    
    Parameters
    ----------
    n_val_samples : integer
    categorical_targets : boolean, default=True
    verbose : integer, default=False

    Returns
    -------
    train_data: tensor dataset 
    val_data: tensor dataset 
    test_data: tensor dataset 

    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train_mean = 0

    x_train, x_test, x_train_mean = preprocess_features(x_train, x_test, 
                                                        substract_pixel_mean=True)
    # else:
    # x_train, x_test = preprocess_features(x_train, x_test, substract_pixel_mean)

    num_classes = len(np.unique(y_train))

    (x_train, y_train), (x_val, y_val) = split_train_dataset(x_train, y_train, n_val_samples)



    if categorical_targets:
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    if verbose:
        print("Shape x_train: ", x_train.shape, "\n",
              "Shape y_train: ", y_train.shape, "\n",
              "Shape x_validation: ", x_val.shape, "\n",
              "Shape y_validation: ", y_val.shape, "\n",
              "Shape x_test: ", x_test.shape, "\n",
              "Shape y_test: ", y_train.shape, "\n")
        
    #train_data, val_data, test_data = convert_to_tensor_dataset(x_train, y_train, 
    #                                                                  x_val, y_val, 
    #                                                                 x_test, y_test)    
    
    # return train_data, val_data, test_data 
    return x_train, y_train, x_val, y_val, x_test, y_test, x_train_mean


def split_train_dataset(x_train, y_train, n_val_samples):
    """
    Parameters
    ----------
    x_train : numpy array
    y_train : numpy array
    n_val_samples : integer
        Number of samples in the validation set.

    Returns
    -------
    (x_train, y_train), (x_val, y_val)    

    """
    indices = np.random.permutation(x_train.shape[0])
    val_idx, train_idx = indices[:n_val_samples], indices[n_val_samples:]
    x_train, x_val = x_train[train_idx, :], x_train[val_idx, :]
    y_train, y_val = y_train[train_idx], y_train[val_idx]
    return (x_train, y_train), (x_val, y_val)
