# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:45:27 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import sys
sys.path.append("..")
import tensorflow.keras
from models.self_att_test import aug_con_2d
from cifar10_dataset.data_loader import get_train_val_test_datasets
import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

validation_size = 5000
batch_size = 128
epochs = 220

if __name__ == "__main__":

    x_train, y_train, x_val, y_val, x_test, y_test, _ = get_train_val_test_datasets(validation_size)

    print("Data loaded.")

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    input_shape = x_train.shape[1:]
    n_colors = x_train.shape[3]
    depth = n_colors * 6 + 2
    #model = resnet34(input_shape, 10)
    #model = ResNet34(10)
    
    #tf.compat.v1.disable_eager_execution()

    model = aug_con_2d(input_shape, 10)

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(),  # Optimizer
                  # Loss function to minimize
                  loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
                  # List of metrics to monitor
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(x_val, y_val))
    scores = model.evaluate(x_test, y_test, batch_size=128)
    # training_module = TrainingEngine(model)
    # training_module.lr_scheduler = StepLearningRate()
    # training_module.optimizer = Adam(lr=training_module.lr_scheduler.get_learning_rate(0))
    # training_module.fit(train_data,
    #                       validation_data,
    #                       batch_size=batch_size,
    #                       epochs=epochs,
    #                       data_augmentation=False)


    # scores = training_module.evaluate(test_data)
    print('Test loss:', scores[1])
    print('Test accuracy:', scores[0])
