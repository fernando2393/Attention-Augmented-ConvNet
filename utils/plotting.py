# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:32:45 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""

import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 


def plot_dataset_examples(x_train, mean, n_rows=10):
    """   
    Plot random samples (number of samples is n_rows*n_rows) of the dataset. 
    
    Parameters
    ----------
    x_train : numpy shape=(n_samples, W, H, 3)
    mean : numpy shape=(W,H,3)
        Train dataset mean substracted to the dataset in order to standardize it.
    n_rows : int, optional
        Number of rows and columns for the grid plot. The default is 10.


    """
    x = x_train.copy()

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
    

def plot_sample(sample, mean, plot_crosses=False, pixel_crosses_x=[], pixel_crosses_y=[]):
    """
    
    Parameters
    ----------
    sample : numpy shape=(W,H,3)
    mean : numpy shape=(W,H,3)
        Train dataset mean substracted to the dataset in order to standardize it.
    plot_crosses : boolean, optional
        Paint crosses of certain pixels or not. The default is False.
    pixel_crosses_x : list, optional
        x coordinate of the pixels you want to pain the cross over. The default is [].
    pixel_crosses_y : list, optional
        y coordinate of the pixels you want to pain the cross over. The default is [].


    """
    x = sample.copy()
    x += mean
    x *= 255
    x = x.astype("uint8")
    plt.axis("off")
    plt.imshow(x)
    if plot_crosses:
        plt.scatter(x = pixel_crosses_x , y = pixel_crosses_y, marker = "x", c = "r")
    plt.show()

    

def plot_attention_map_pixel(attention_maps, pixel_coord, axes, idx_col_subplt):
    """
    
    Parameters
    ----------
    attention_maps : tensors shape=[1, N_h, W, H, W, H]
        DESCRIPTION.
    pixel_coord : tuple
        DESCRIPTION.
    axes : numpy array 
        Axes of the plot grid.
    idx_col_subplt : int

    Returns
    -------
    axes_subplt : TYPE
        DESCRIPTION.

    """
    N_h = tf.shape(attention_maps)[1].numpy()

    attent_map_pixel = attention_maps[:,:, pixel_coord[0], pixel_coord[1]]
    
    for idx_head in range(N_h):  
        att_map_head = attent_map_pixel[0, idx_head]
        axes[idx_head][idx_col_subplt].imshow(att_map_head, cmap='Oranges')
        axes[idx_head][idx_col_subplt].set_axis_off()
    return axes


def plot_attention_maps_pixels(attention_maps, pixels_x, pixels_y):
    """
    Plot a subplot where each column is the self attention map of each pixel and
    the rows represent the self attention map of each head of the MHA.

    Parameters
    ----------
    attention_maps : tensors shape=[1, N_h, W, H, W, H]
        Attention map of an image.
    pixels_x : list
        x coordinate of the pixels we want to plot their attention map.
    pixels_y : list
        y coordinate of the pixels we want to plot their attention map.

    """
    if len(pixels_x) != len(pixels_y):
        raise ValueError("The length of pixels_x and pixels_y should be the same")
    n_points = len(pixels_x)
    N_h = tf.shape(attention_maps)[1].numpy()
    fig, axes = plt.subplots(nrows=N_h, ncols=n_points)
    for idx_pixel in range(n_points):
        pixel_coord = (pixels_y[idx_pixel], pixels_x[idx_pixel])
        axes = plot_attention_map_pixel(attention_maps, pixel_coord, axes, idx_pixel)
    
    ## Set axis labels of the subplot
    # axes = add_axis_labels(axes, N_h, n_points)
    
    plt.ylabel("Pixels")
    plt.xlabel("Heads")
    plt.show()

"""
def add_axis_labels(axes_subplt, N_h, n_points):
    
    # set rows titles 
    for idx_head in range(N_h):
        title_rows = "head " + str(idx_head)
        axes_subplt[0,idx_head].ylabel(title_rows)
    
    # set rows titles 
    for idx_point in range(n_points): 
        title_cols = "pixel " + str(idx_point+1)
        axes_subplt[idx_point,0].title(title_cols)
    return axes_subplt
        
"""
        