# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:06:07 2020

@author: FlaviaGV
"""

import tensorflow as tf 


def get_img_shape(x):
    """
    Return list of dims, statically when possible, if not dynamically.

    Parameters
    ----------
    x : tensor/s

    Returns
    -------
    ret : list
        [n_samples_batch, height, width, depth]

    """
    static_dims = x.get_shape().as_list()
    dynamic_shape = tf.shape(x)
    x_shape = []
    for i, static_dim in enumerate(static_dims):
        dim = static_dim or dynamic_shape[i]
        x_shape.append(dim)
    return x_shape


def rel_to_abs(x):
    """Converts tensor from relative to absolute indexing."""
    # [B, Nh, L, 2L−1]
    B, Nh, L, _= get_img_shape(x)
    # Pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((B, N_h, L, 1))
    x = tf.concat([x, col_pad], axis=3)
    flat_x = tf.reshape(x, [B, N_h, L ∗ 2 ∗ L])
    flat_pad = tf.zeros((B, N_h, L−1))
    flat x padded = tf.concat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [B, N_h, L+1, 2∗L−1])
    final_x = final_x[:, :, :L, L−1:]
    return final_x