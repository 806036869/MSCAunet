# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:09:44 2021

@author: angelou
"""
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, add
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras import backend as K

import tensorflow as tf
from train_predict_test.caculate_params import stats_graph, load_pb
from tensorflow.python.framework import graph_util

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def DCBlock(U, inp, alpha=1.67):
    '''
    DC Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    # shortcut = inp

    # shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
    #                      int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3_1 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                          activation='relu', padding='same')

    conv5x5_1 = conv2d_bn(conv3x3_1, int(W * 0.333), 3, 3,
                          activation='relu', padding='same')

    conv7x7_1 = conv2d_bn(conv5x5_1, int(W * 0.5), 3, 3,
                          activation='relu', padding='same')

    out1 = concatenate([conv3x3_1, conv5x5_1, conv7x7_1], axis=3)
    out1 = BatchNormalization(axis=3)(out1)

    conv3x3_2 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                          activation='relu', padding='same')

    conv5x5_2 = conv2d_bn(conv3x3_2, int(W * 0.333), 3, 3,
                          activation='relu', padding='same')

    conv7x7_2 = conv2d_bn(conv5x5_2, int(W * 0.5), 3, 3,
                          activation='relu', padding='same')
    out2 = concatenate([conv3x3_2, conv5x5_2, conv7x7_2], axis=3)
    out2 = BatchNormalization(axis=3)(out2)

    out = add([out1, out2])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def DCUNet(num_classes, input_shape, lr_init, lr_decay):
    '''
    DC-UNet

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''
    inputs = Input(input_shape)

    dcblock1 = DCBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dcblock1)
    dcblock1 = ResPath(32, 4, dcblock1)

    dcblock2 = DCBlock(32 * 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dcblock2)
    dcblock2 = ResPath(32 * 2, 3, dcblock2)

    dcblock3 = DCBlock(32 * 4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dcblock3)
    dcblock3 = ResPath(32 * 4, 2, dcblock3)

    dcblock4 = DCBlock(32 * 8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dcblock4)
    dcblock4 = ResPath(32 * 8, 1, dcblock4)

    dcblock5 = DCBlock(32 * 16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32 * 8, (2, 2), strides=(2, 2), padding='same')(dcblock5), dcblock4], axis=3)
    dcblock6 = DCBlock(32 * 8, up6)

    up7 = concatenate([Conv2DTranspose(
        32 * 4, (2, 2), strides=(2, 2), padding='same')(dcblock6), dcblock3], axis=3)
    dcblock7 = DCBlock(32 * 4, up7)

    up8 = concatenate([Conv2DTranspose(
        32 * 2, (2, 2), strides=(2, 2), padding='same')(dcblock7), dcblock2], axis=3)
    dcblock8 = DCBlock(32 * 2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(dcblock8), dcblock1], axis=3)
    dcblock9 = DCBlock(32, up9)

    x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(dcblock9)

    model = Model(inputs, x)
    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss=['binary_crossentropy'],
                  metrics=[dice_coef])
    # metrics='binary_accuracy')
    return model