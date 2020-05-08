# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:17:23 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import tensorflow as tf
import utils
from tensorflow.keras.layers import Conv2D


class SelfAttention2D:

    def __init__(self, N_h, depth_k, depth_v, relative=True):
        """
        Parameters
        ----------
        N_h : integer
            number of heads.
        depth_k : integer
            keys depth (all attention heads together). Same value for querys depth.
        depth_v_h : integer
            values depth (all attention heads together). 
        relative : bool, optional
        """
        self.N_h = N_h
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.depth_k_h = self.depth_k // self.N_h
        self.depth_v_h = self.depth_v // self.N_h
        self.relative = relative

    
    def forward_pass(self, inputs): # self_attention_2d
        """
        2d relative self−attention.

        Parameters
        ----------
        inputs : image tensor/s shape=[n_samples_batch, H, W, 3]

        Returns
        -------
        MHA_output: tensor/s shape=[n_samples_batch, H, W, self.depth_v]
          Multi-head attention features. 

        """
        _, H, W, _ = utils.get_img_shape(inputs)

        flatten_hw = lambda x, d: tf.reshape(x, [-1, self.N_h, H*W, d])

        # Compute q, k, v
        total_depth = 2 * self.depth_k + self.depth_v
        k_q_v = Conv2D(filters = total_depth, kernel_size = 1)(inputs) # point-wise convolution
        k, q, v = tf.split(k_q_v, [self.depth_k, self.depth_k, self.depth_v], axis=3)
        q *= self.depth_k_h ** -0.5 # scaled dot−product
        
        # After splitting, shape is [B, N_h, H, W, d_k_h or d_v_h]
        q_h = self.split_heads(q)
        k_h = self.split_heads(k)
        v_h = self.split_heads(v)

        # [B, Nh, HW, HW]
        logits = tf.matmul(flatten_hw(q_h, self.depth_k_h), flatten_hw(k_h, self.depth_k_h), transpose_b=True)

        if self.relative:
            rel_logits_h, rel_logits_w = self.relative_logits(q_h, H, W)
            logits += rel_logits_h
            logits += rel_logits_w
        
        # logits_scaled = logits/(self.depth_k_h ** -0.5)
        # Attention map
        weights = tf.nn.softmax(logits)
        O_h = tf.matmul(weights, flatten_hw(v_h, self.depth_v_h))
        O_h = tf.reshape(O_h, [-1, self.N_h, H, W, self.depth_v_h])
        O = self.combine_heads_2d(O_h)

        # Project heads
        MHA_output = Conv2D(filters=self.depth_v, kernel_size=1)(O) # point-wise product

        return MHA_output


    def split_heads(self, inputs):
        """
        Split channels into multiple heads.

        Parameters
        ----------
        inputs : tensor/s with shape=[n_samples_batch, height, width, total_depth]

        Returns
        -------
        inputs_h: tensor/s with shape=[n_samples_batch, n_heads, height, width, depth_per_head]        
        """
        B, H, W, d = utils.get_img_shape(inputs)
        ret_shape = [B, H, W, self.N_h, d//self.N_h]
        split = tf.reshape(inputs, ret_shape)
        inputs_h = tf.transpose(split, [0, 3, 1, 2, 4])

        return inputs_h


    def combine_heads_2d(self, inputs_h):
        """
        Combine heads (inverse of split heads 2d).
        
        Parameters
        ----------
        inputs_h : tensor/s with shape=[n_samples_batch, n_heads, height, width, depth_per_head] 

        Returns
        -------
        inputs: tensor/s with shape=[n_samples_batch, height, width, total_depth]
        """
        transposed = tf.transpose(inputs_h, [0, 2, 3, 1, 4])
        _, channels = utils.get_img_shape(transposed)[-2:]
        ret_shape = utils.get_img_shape(transposed)[:-2] + [self.N_h * channels]
        inputs = tf.reshape(transposed, ret_shape)
        return inputs
    
    
    def relative_logits(self, q_h, H, W):
        """
        Compute relative logits.
        
        Parameters
        ----------
        q_h : tensors shape=[n_samples_batch, self.N_h, W, H, self.depth_q_h]
          queries per head 
        H: int
          height of the image
        W: int
          width of the image 

        Returns
        -------
        rel_logits_h:
        rel_logits_w:
        """
        # Relative logits in width dimension first.
        rel_embedding_w = tf.compat.v1.get_variable('r_width', shape=(2*W-1, self.depth_k_h), 
                                          initializer = tf.random_normal_initializer(self.depth_k_h**-0.5))         # TF 2.0 is tf.Variable 
        
        
        # [B, N_h, HW, HW]
        rel_logits_w = self.relative_logits_1d(q_h, rel_embedding_w, H, W, [0, 1, 2, 4, 3, 5])
        
        # Relative logits in height dimension next.
        # For ease, we:
        # 1) transpose height and width
        # 2) repeat the above steps and
        # 3) transpose to eventually put the logits in their right positions.
        rel_embeddings_h = tf.compat.v1.get_variable('r_height', shape=(2*H-1, self.depth_k_h),
                                           initializer=tf.random_normal_initializer(self.depth_k_h**-0.5))
        # [B, Nh, HW, HW]
        rel_logits_h = self.relative_logits_1d(tf.transpose(q_h, [0, 1, 3, 2, 4]), rel_embeddings_h, W, H, [0, 1, 4, 2, 5, 3])
        
        return rel_logits_h, rel_logits_w
    
    
    def relative_logits_1d(self, q_h, rel_k, H, W, transpose_mask):
        """
        Compute relative logits along one dimenion.
        
        Parameters
        ----------
        q_h: tensors shape=[n_samples_batch, self.N_h, W, H, self.depth_q_h]
            queries per head 
        rel_k:
            
        H: int
          height of the image
        W: int 
          height of the image
        transpose_mask: list    
            indicates indexes for transposing
          
        Returns
        -------
        rel_logits:
            
        """
        # 'tf.einsum' is to define the dimensions of the input and the output 
        # after a matrix multiplication. So automatically it detects the type of 
        # matrix multiplication that it would be. 
        rel_logits = tf.einsum('bhxyd,md->bhxym', q_h, rel_k) 
        
        # Collapse height and heads
        rel_logits = tf.reshape(rel_logits, [-1, self.N_h*H, W, 2*W-1])
        rel_logits = utils.rel_to_abs(rel_logits)
        
        # Shape it and tile height times
        rel_logits = tf.reshape(rel_logits, [-1, self.N_h, H, W, W])
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
        
        # Reshape for adding to the logits.
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1, self.N_h, H*W, H*W])
        
        return rel_logits




"""
import sys 

sys.path.insert(1, "..")
from cifar10_dataset import data_loader

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_train_val_test_data(5000)

#x_train = tf.data.Dataset.from_tensor_slices(x_train)
a = tf.convert_to_tensor(x_train[:5])
attention_layer = SelfAttention2D(N_h=8, depth_k=160, depth_v=160)
attention_layer.forward_pass(a)

"""