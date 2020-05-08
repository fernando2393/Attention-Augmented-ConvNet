# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:30:12 2020

@author: matte
"""
import tensorflow as tf
import tensorflow.keras
from preprocessing.augmentation import Augment2D
from engine.learning_rate.linear_cos_annealing import LinearCosAnnelingLrSchedule


class TrainingEngine:
    def __init__(self, model):
        
        self.model = model
        
        self.lr_scheduler = LinearCosAnnelingLrSchedule(lr_linear_final_epoch=25)

        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_scheduler.get_learning_rate(0))
    
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        
        self.data_augmentation_module = Augment2D(True, True, True, zero_padding_shape=(40, 40), random_crop_shape=(32, 32))
        
        
    @tf.function
    def __train_step(self, images, labels):
      with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=True)
        loss = self.loss_object(labels, predictions)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
      
      
    @tf.function
    def __test_step(self, images, labels):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = self.model(images, training=False)
      t_loss = self.loss_object(labels, predictions)
    
      self.test_loss(t_loss)
      self.test_accuracy(labels, predictions)
      
    def fit(self, train_data, validation_data, batch_size=100, epochs=20, shuffle=True, data_augmentation=False, verbose=True):

      for epoch in range(epochs):
          self.train_loss.reset_states()
          self.train_accuracy.reset_states()
          self.test_loss.reset_states()
          self.test_accuracy.reset_states()
          
          self.optimizer.lr.assign(self.lr_scheduler.get_learning_rate(epoch))
  
          if shuffle:
              epoch_train_data = train_data.shuffle(len(list(train_data)))
          else:
              epoch_train_data = train_data
          if data_augmentation:
              epoch_train_data = self.data_augmentation_module.transform(epoch_train_data)
              
          batched_train_data = epoch_train_data.batch(batch_size)
          for batch_x, batch_y in batched_train_data:
              self.__train_step(batch_x, batch_y)

          batched_val_data = epoch_train_data.batch(batch_size)
          for batch_x_val, batch_y_val in batched_val_data:
              self.__test_step(batch_x_val, batch_y_val)
          
          if verbose:
              template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}, Learning rate: {}'
              print(template.format(epoch + 1,
                                    self.train_loss.result(),
                                    self.train_accuracy.result() * 100,
                                    self.test_loss.result(),
                                    self.test_accuracy.result() * 100,
                                    self.optimizer.lr.last_lr))