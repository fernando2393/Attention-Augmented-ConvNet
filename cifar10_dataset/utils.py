# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:58:02 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_dataset_examples(x_train, mean=None, n_rows=10):
    x = x_train

    if mean is not None:
        x += mean
        x *= 255

    x = x.astype("uint8")
    fig, axes1 = plt.subplots(n_rows, n_rows, figsize=(n_rows, n_rows))
    for j in range(n_rows):
        for k in range(n_rows):
            i = np.random.choice(range(len(x)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(x[i:i+1][0])
    plt.show()


def convert_to_tensor_dataset(x_train, y_train, x_val, y_val, x_test, y_test):
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, validation_data, test_data