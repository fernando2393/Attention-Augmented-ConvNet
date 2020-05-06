# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:17:23 2020

@author: flaviagv
"""

import tensorflow as tf
import utils
from tensorflow.keras.layers import Conv2D

import sys 

sys.path.insert(1, "..")
from cifar10_dataset import data_loader


class SelfAttention2D:

    def __init__(self, N_h, depth_k, depth_v, relative=True):
        """
        Parameters
        ----------
        depth_k : integer
            keys depth (all attention heads together). Same value for querys depth.
        depth_v_h : integer
            values depth (all attention heads together).
        N_h : integer
            number of heads.
        relative : bool, optional
            DESCRIPTION. The default is True.

        """
        self.N_h = N_h
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.depth_k_h = self.depth_k // self.N_h
        self.depth_v_h = self.depth_v // self.N_h
        self.relative = relative

    

    def split_heads(self, inputs):
        """Split channels into multiple heads."""
        B, H, W, d = utils.get_img_shape(inputs)
        ret_shape = [B, H, W, self.N_h, d//self.N_h]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0, 3, 1, 2, 4])


    def combine_heads(self, inputs):
        """Combine heads (inverse of split heads 2d)."""
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        Nh, channels = utils.get_img_shape(transposed)[-2:]
        ret_shape = utils.get_img_shape(transposed)[:-2] + [self.N_h * channels]
        return tf.reshape(transposed, ret_shape)


    def self_attention_2d(self, inputs):
        """
        2d relative self−attention.

        Parameters
        ----------
        inputs : image tensor/s
            DESCRIPTION.

        Returns
        -------
        None.

        """
        _, H, W, _ = utils.get_img_shape(inputs)

        flatten_hw = lambda x, d: tf.reshape(x, [-1, self.N_h, H*W, d])

        # Compute q, k, v
        total_depth = 2 * self.depth_k + self.depth_v
        k_q_v = Conv2D(inputs, filters = total_depth, kernel_size = 1) # point-wise convolution
        k, q, v = tf.split(k_q_v, [self.depth_k, self.depth_k, self.depth_v], axis=3)
        q *= self.depth_k_h ** -0.5 # scaled dot−product
        # After splitting, shape is [B, N_h, H, W, d_k_h or d_v_h]
        q_h = self.split_heads(q, self.N_h)
        k_h = self.split_heads(k, self.N_h)
        v_h = self.split_heads(v, self.N_h)

        # [B, Nh, HW, HW]
        logits = tf.matmul(flatten_hw(q_h, self.depth_k_h), flatten_hw(k_h, self.depth_k_h), transpose_b=True)


        # if self.relative:
        #     rel logits h, rel logits w = relative logits(q, H, W, Nh,
        #     dkh)
        #     logits += rel_logits_h
        #     logits += rel_logits_w

        # Attention map
        weights = tf.nn.softmax(logits)
        O_h = tf.matmul(weights, flatten_hw(v_h, self.depth_v_h))
        O_h = tf.reshape(O_h, [-1, self.N_h, H, W, self.depth_v_h])
        O = self.combine_heads_2d(O_h)

        # Project heads
        MHA_output = tf.layers.conv2d(O, filters =  self.depth_v, kernel_size=1) # point-wise product

        return MHA_output

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_train_val_test_data(5000)

#x_train = tf.data.Dataset.from_tensor_slices(x_train)
a = tf.convert_to_tensor(x_train)
attention_layer = SelfAttention2D(N_h=8, depth_k=20, depth_v=20)
attention_layer.self_attention_2d(a)

