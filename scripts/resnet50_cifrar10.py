# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:19:21 2020

@author: matte
"""

import sys
sys.path.append("..")

import tensorflow.keras
from models.resnet50 import resnet50_v1
from cifar10_dataset.data_loader import get_train_val_test_data
import numpy as np
from preprocessing.augmentation import Augment2D
import tensorflow as tf

validation_size = 5000
batch_size = 128 
epochs = 500
num_classes = 10

class MyLrSchedule():
    def __init__(self, cos_lr_n_min=0.004, cos_lr_n_max=0.020, cos_lr_T_0=10, cos_lr_T_mult=2, lr_linear_final_epoch=25, batch_size=128):
        self.cos_lr_n_min = cos_lr_n_min
        self.cos_lr_n_max = cos_lr_n_max
        self.cos_lr_T_0 = cos_lr_T_0
        self.cos_lr_T_mult = cos_lr_T_mult
        self.lr_linear_final_epoch = lr_linear_final_epoch
        self.batch_size = batch_size

        self.last_lr = 0.0

        self.__cos_lr_passed_epochs = 0 
        self.__cos_lr_cycles = 0
  
    def get_learning_rate(self, epoch):
      if epoch <= self.lr_linear_final_epoch:
        lr = 0.2 * self.batch_size / 256 * (epoch + 1) / 25
        self.__cos_lr_passed_epochs = epoch + 1
      else:
        T_curr = epoch - self.__cos_lr_passed_epochs
        T_i = self.cos_lr_T_mult**(self.__cos_lr_cycles) * self.cos_lr_T_0
        if T_curr == T_i:
            self.__cos_lr_cycles += 1
            self.__cos_lr_passed_epochs = epoch + 1
            
        lr = self.cos_lr_n_min + 0.5 * (self.cos_lr_n_max - self.cos_lr_n_min) * (1 + np.cos(T_curr/(T_i) * np.pi))
      return lr

@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels, model, loss_object, test_loss, test_accuracy):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

def train_network(train_data, validation_data, model,  batch_size=100, epochs=20, shuffle=True, data_augmentation=False):
    
    lr_scheduler = MyLrSchedule(lr_linear_final_epoch=25)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler.get_learning_rate(0))
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    data_augmentation_engine = Augment2D(True, True, True, zero_padding_shape=(40, 40), random_crop_shape=(32, 32))
    
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        optimizer.lr.assign(lr_scheduler.get_learning_rate(epoch))

        if shuffle:
            epoch_train_data = train_data.shuffle(len(list(train_data)))
        else:
            epoch_train_data = train_data
        if data_augmentation:
            epoch_train_data = data_augmentation_engine.transform(epoch_train_data)
            
        batched_train_data = epoch_train_data.batch(batch_size)
        for batch_x, batch_y in batched_train_data:
            #x_batch, y_batch = batch
            train_step(batch_x, batch_y, model, loss_object, optimizer, train_loss, train_accuracy)
        '''
        val_len = len(list(validation_data))
        x_val, y_val = next(iter(validation_data.batch(val_len)))
        test_step(x_val, y_val, model, loss_object, test_loss, )
        '''
        batched_val_data = epoch_train_data.batch(batch_size)
        for batch in batched_val_data:
            test_step(x_val, y_val, model, loss_object, test_loss, test_accuracy)
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == "__main__":
        
    (x_train, y_train), (x_val, y_val), (x_test, y_test), _ = get_train_val_test_data(validation_size)
    print("Data loaded.")
    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    input_shape = x_train.shape[1:]
    
    n_colors = x_train.shape[3]

    depth = n_colors * 6 + 2
    
    model = resnet50_v1(input_shape=input_shape, depth=depth)

    train_network(train_data,
                          validation_data,
                          model,
                          batch_size=batch_size,
                          epochs=epochs,
                          data_augmentation=True)
    
    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])