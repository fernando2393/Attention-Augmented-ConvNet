# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:58:02 2020

@author: Matosevic
"""

import matplotlib.pyplot as plt
import numpy as np 

import keras.layers

def plot_dataset_examples(x_train, n_rows=10):
  x = x_train.astype("uint8")

  fig, axes1 = plt.subplots(n_rows, n_rows,figsize=(n_rows,n_rows))
  for j in range(n_rows):
      for k in range(n_rows):
          i = np.random.choice(range(len(x)))
          axes1[j][k].set_axis_off()
          axes1[j][k].imshow(x[i:i+1][0])