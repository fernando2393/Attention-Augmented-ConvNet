# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:06:06 2020

@author: matte
"""

class StepLearningRate():
    
    def get_learning_rate(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        #print('Learning rate: ', lr)
        return lr