# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:14:26 2020

@author: matte
"""

import tensorflow as tf
from layers.test_layer import augmented_conv2d


def layer_selection(augmented_cnt, not_augmented_layers, inputs, filters, kernel_size, strides):
    if augmented_cnt < not_augmented_layers:
        x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, use_bias=True,
                                   padding='same')(inputs)
    else:
        v = 0.1
        x = augmented_conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, k=2*v, v=v)

    augmented_cnt += 1

    return x, augmented_cnt


def wide_basic(inputs, in_planes, out_planes, stride):
    if stride != 1 or in_planes != out_planes:
        skip_c = augmented_conv2d(inputs, out_planes, kernel_size=1, strides=stride, k=1.0, v=1.0)
    # elif stride != 1 or in_planes != out_planes:
    #     skip_c = tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=True,
    #                                     padding='same')(inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(inputs)
    x = tf.nn.relu(x)
    x = augmented_conv2d(x, out_planes, kernel_size=3, strides=1, k=1.0, v=1.0)
    #x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=1, use_bias=True, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=0.1, trainable=True)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True,
                                           trainable=True)(x)
    x = tf.nn.relu(x)
    x = augmented_conv2d(x, out_planes, kernel_size=3, strides=stride, k=1.0, v=1.0)
    #x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, use_bias=True, padding='same')(x)

    x = tf.add(skip_c, x)

    return x


def wide_layer(out, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    i = 0
    for strid in strides:
        out = wide_basic(out, in_planes, out_planes, strid)
        in_planes = out_planes
        i += 1

    return out


def make_resnet_fully_selfatt_filter(inputs, depth=28, widen_factor=10, num_classes=10, augmented=False):
    n = (depth - 4) / 6
    k = widen_factor
    print('| Wide-Resnet %dx%d' % (depth, k))
    nstages = [16, 16 * k, 32 * k, 64 * k]
    #x = tf.keras.layers.Conv2D(nstages[0], kernel_size=3, strides=1, use_bias=True, padding='same')(inputs)
    x = augmented_conv2d(inputs, nstages[0], kernel_size=3, strides=1, k=1.0, v=1.0)
    x = wide_layer(x, nstages[0], nstages[1], n, stride=1)
    x = wide_layer(x, nstages[1], nstages[2], n, stride=2)
    x = wide_layer(x, nstages[2], nstages[3], n, stride=2)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
