from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import cifar10_dataset.data_loader as dt_ld
import cifar10_dataset.utils as ut

BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
EPOCHS = 200  # 200
USE_AUGMENTATION = True
NUM_CLASSES = 10
COLORS = 3
VERSION = 1
if VERSION == 1:  # ResNet version depth config
    DEPTH = COLORS * 6 + 2
else:
    DEPTH = COLORS * 9 + 2


def lr_decay(epoch, balanced=True, decay_epoch=None):
    eta = 1e-3
    if balanced:
        decay_epoch = list(np.linspace(0, EPOCHS, 5))
        decay_epoch.reverse()
    for dc in decay_epoch:
        if epoch <= dc:
            exp = int(EPOCHS / dc)
            eta = pow(10, -exp)
            break

    return eta


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_norm=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size, strides, padding='same', kernel_size='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x


def main():
    lr_decay(20)
    train, val, test, x_mean = dt_ld.get_train_val_test_data(5000)
    ut.plot_dataset_examples(train[0], x_mean)


if __name__ == "__main__":
    main()



