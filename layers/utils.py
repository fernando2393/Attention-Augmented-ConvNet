# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:06:07 2020

@author: FlaviaGV
"""

import tensorflow as tf 


def get_img_shape(x):
    """
    TODO: put is as part of utils??
    Return list of dims, statically when possible, if not dynamically.

    Parameters
    ----------
    x : tensor/s

    Returns
    -------
    ret : list
        [n_samples, height, width, depth]

    """
    static_dims = x.get_shape().as_list()
    dynamic_shape = tf.shape(x)
    x_shape = []
    for i, static_dim in enumerate(static_dims):
        dim = static_dim or dynamic_shape[i]
        x_shape.append(dim)
    return x_shape