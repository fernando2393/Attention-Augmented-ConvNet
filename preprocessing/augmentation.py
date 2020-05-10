# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:19:12 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Augment2D:

    def __init__(self, horizontal_flip, zero_padding, random_crop,
                 zero_padding_shape=(0, 0), random_crop_shape=(0, 0)):

        self.horizontal_flip = horizontal_flip
        self.zero_padding = zero_padding
        self.zero_padding_shape = zero_padding_shape  # (height, width)
        self.random_crop = random_crop
        self.random_crop_shape = random_crop_shape  # (height, width)

    def transform(self, data):
        """
        Apply the following transformations to the data if they are activated:
        flip horizontally with probability 0.5, add zero padding with the image in the center and 
        crop the image randomly. 
        
        Parameters
        ----------
        data : numpy array 

        Returns
        -------
        data : numpy array 

        """
        if self.horizontal_flip:
            data = data.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                            num_parallel_calls=AUTOTUNE)
        if self.zero_padding:
            data = data.map(lambda x, y: (tf.image.resize_with_crop_or_pad(x,
                                                                       self.zero_padding_shape[0],
                                                                       self.zero_padding_shape[1]), y),
                            num_parallel_calls=AUTOTUNE)
        if self.random_crop:
            size = [self.random_crop_shape[0], self.random_crop_shape[1], 3]
            data = data.map(lambda x, y: (tf.image.random_crop(x, size, seed=None, name=None), y),
                            num_parallel_calls=AUTOTUNE)

        return data
