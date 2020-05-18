# -*- coding: utf-8 -*-
"""
Created on Sat May  11 2:29:36 2020

@author: MatteoDM, FernandoGS, FlaviaGV
"""
import math


class WideLearningRate:

    @staticmethod
    def get_learning_rate(epoch):
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
        optim_factor = 0
        init = 0.1
        if epoch > 90:
            optim_factor = 3
        elif epoch > 60:
            optim_factor = 2
        elif epoch > 30:
            optim_factor = 1

        return init * math.pow(0.2, optim_factor)
