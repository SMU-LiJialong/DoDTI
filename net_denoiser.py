#!/usr/bin/env python
# encoding: utf-8

#****************************
# the DnCNN denoiser used in AuxVarLayer
#***************************

import tensorflow as tf
import tensorflow_addons as tfa

class DnCNN(tf.keras.layers.Layer):
    def __init__(self, Nblock, Nfilters, bn_flag=False, f=3, name=None):
        super(DnCNN, self).__init__(name=name)
        self.Nconv = Nblock
        self.conv = {}  # convolution
        for i in range(1, self.Nconv):
            self.conv[i] = tf.keras.layers.Conv3D(filters=Nfilters, kernel_size=(f, f, f), strides=(1, 1, 1),
                                                  padding='same', kernel_initializer='he_normal', name='conv' + str(i))

        # self.conv_out = tf.keras.layers.Conv3D(filters=7, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
        #                                        kernel_initializer='he_normal', name='conv_out')
        self.conv_out1 = tf.keras.layers.Conv3D(filters=1, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
                                                kernel_initializer='he_normal', name='conv_out1')
        self.conv_out2 = tf.keras.layers.Conv3D(filters=3, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
                                                kernel_initializer='he_normal', name='conv_out2')
        self.conv_out3 = tf.keras.layers.Conv3D(filters=3, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
                                                kernel_initializer='he_normal', name='conv_out3')
        self.bn_flag = bn_flag
        self.BN = {}
        for j in range(1, self.Nconv):
            self.BN[j] = tf.keras.layers.BatchNormalization(name='BN'+str(j))

        self.GN = {}
        for j in range(1, self.Nconv):
            self.GN[j] = tfa.layers.GroupNormalization(name='GN'+str(j))

    def call(self, inputs):
        x = inputs
        # x = self.GN[1](x)                 # before input denoiser, normalize x
        x = self.conv[1](x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(2, self.Nconv):
            x = self.conv[i](x)
            if self.bn_flag is True:
                x = self.BN[i](x)
                # x = self.GN[i](x)
            x = tf.keras.layers.ReLU()(x)
        # res = self.conv_out(x)
        x1 = self.conv_out1(x)
        x2 = self.conv_out2(x)
        x3 = self.conv_out3(x)
        res = tf.keras.layers.Concatenate()([x1, x2, x3])

        # %% only output denoised parameter image, for train
        denoised = tf.math.subtract(inputs, res)  # denoised parameter
        return denoised
