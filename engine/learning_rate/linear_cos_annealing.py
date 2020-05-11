# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:21:32 2020

@author: matte
"""
import numpy as np

class LinearCosAnnelingLrSchedule():
    ''
    def __init__(self, cos_lr_n_min=0.004, cos_lr_n_max=0.020, cos_lr_T_0=10, cos_lr_T_mult=2, lr_linear_final_epoch=25, batch_size=128):
        
        self.cos_lr_n_min = cos_lr_n_min
        self.cos_lr_n_max = cos_lr_n_max
        self.cos_lr_T_0 = cos_lr_T_0
        self.cos_lr_T_mult = cos_lr_T_mult
        self.lr_linear_final_epoch = lr_linear_final_epoch
        self.batch_size = batch_size
        
        self.__cos_lr_passed_epochs = 0 
        self.__cos_lr_cycles = 0
        self.last_lr = 0
    
    
    def get_learning_rate(self, epoch):
        """
        Parameters
        ----------
        epoch : integer
            Current epoch.

        Returns
        -------
        lr : float
            Current learning rate per epoch.

        """
        if epoch <= self.lr_linear_final_epoch:
            lr = 0.2 * self.batch_size / 256 * (epoch + 1) / 25
            self.__cos_lr_passed_epochs = epoch + 1
        else:
          T_curr = epoch - self.__cos_lr_passed_epochs
          T_i = self.cos_lr_T_mult**(self.__cos_lr_cycles) * self.cos_lr_T_0
          if T_curr == T_i:
              self.__cos_lr_cycles += 1
              self.__cos_lr_passed_epochs = epoch + 1
              
          lr = self.cos_lr_n_min + 0.5 * (self.cos_lr_n_max - self.cos_lr_n_min) * (1 + np.cos(T_curr/(T_i) * np.pi))
          
        self.last_lr = lr
        return lr