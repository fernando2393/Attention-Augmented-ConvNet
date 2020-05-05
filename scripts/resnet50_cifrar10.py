# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:34:40 2020

@author: matte
"""
import sys
sys.path.append("..")

import tensorflow.keras
from tensorflow.keras.callbacks import LearningRateScheduler
from models.resnet50 import resnet50_v1
from cifar10_dataset.data_loader import get_train_val_test_data
import numpy as np
import sklearn
from preprocessing.augmentation import Augment2D
from tqdm import tqdm
import tensorflow as tf


batch_size = 128 
epochs = 500
num_classes = 10
validation_size = 5000
curr_epoch = 0

__cos_lr_passed_epochs = 0 
__cos_lr_cycles = 0

cos_lr_n_min = 0.004
cos_lr_n_max = 0.020
cos_lr_T_0 = 10
cos_lr_T_mult = 2
lr_linear_final_epoch = 25

def lr_schedule(epoch):
    """Learning Rate Schedule

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    global __cos_lr_cycles
    global __cos_lr_passed_epochs
    
    if curr_epoch <= lr_linear_final_epoch:
      lr = 0.2 * batch_size / 256 * (curr_epoch+1) / 25
      global __cos_lr_passed_epochs
      __cos_lr_passed_epochs = curr_epoch + 1
      
    #lr = 1e-3
    else:
        T_curr = curr_epoch - __cos_lr_passed_epochs
        T_i = cos_lr_T_mult**(__cos_lr_cycles) * cos_lr_T_0
        if T_curr == T_i:
            __cos_lr_cycles += 1
            __cos_lr_passed_epochs = curr_epoch + 1
            
        lr = cos_lr_n_min + 0.5 * (cos_lr_n_max - cos_lr_n_min) * (1 + np.cos(T_curr/(T_i) * np.pi))
        print('Epoch ', curr_epoch, 'Learning rate: ', lr, 'T_curr ', T_curr, 'T_i ', T_i)
    return lr


def train_network(x_train, y_train, batch_size=100, epochs=20, validation_data=(), shuffle=True, data_augmentation=False):
    
    input_shape = x_train.shape[1:]
    
    n_colors = x_train.shape[3]

    depth = n_colors * 6 + 2
    
    model = resnet50_v1(input_shape=input_shape, depth=depth)
    
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    input_dim = x_train.shape[0]
    
    global curr_epoch
    curr_epoch = 0
    
    for epoch in tqdm(range(epochs)):
        curr_epoch = epoch
        if shuffle:
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
            for batch_idx in range(batch_size):
                j_start = batch_idx* batch_size
                j_end = (batch_idx + 1) * batch_size
                if batch_idx == batch_size - 1:
                    j_end = input_dim
                x_batch = x_train[j_start:j_end, :]
                y_batch = y_train[j_start:j_end, :]
                
                if data_augmentation:
                    data_augmentor = Augment2D(True, True, True, zero_padding_shape=(40, 40), random_crop_shape=(32, 32))
                    tensor_x_batch = tf.data.Dataset.from_tensor_slices(x_batch)
                    x_batch = data_augmentor.transform(tensor_x_batch)
                model.fit(x_batch, y_batch,
                                  batch_size=batch_size,
                                  epochs=1,
                                  validation_data=(validation_data[0], validation_data[1]),
                                  shuffle=False,
                                  callbacks=[lr_scheduler], verbose=0)
    return model

if __name__ == "__main__":
    
    for e in range(epochs):
        curr_epoch = e 
        lr_schedule(e)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_val_test_data(validation_size)
    '''
    input_shape = x_train.shape[1:]
    
    ##TODO: Add data augmentation
    
    
    n_colors = x_train.shape[3]

    depth = n_colors * 6 + 2
    
    model = resnet50_v1(input_shape=input_shape, depth=depth)
    
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[lr_scheduler])
    '''
    # model = train_network(x_train, y_train,
    #                       batch_size=batch_size,
    #                       epochs=epochs,
    #                       validation_data=(x_val, y_val),
    #                       data_augmentation=True)
    
    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])