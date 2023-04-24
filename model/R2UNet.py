#coding=utf-8
from keras.models import *
from keras.layers import *
from keras.optimizers import adam_v2
from keras import backend as K

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Recurrent_block(input, channel, t=2):
    for i in range(t):
        if i == 0:
            x = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        out = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(add([x, x]))
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
    return out

def RRCNN_block(input, channel, t=2):
    x1 = Conv2D(channel, kernel_size=(1, 1), strides=1, padding='same')(input)
    x2 = Recurrent_block(x1, channel, t=t)
    x2 = Recurrent_block(x2, channel, t=t)
    out = add([x1, x2])
    return out

def R2UNet(num_classes, input_shape, lr_init, lr_decay):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    inputs = Input(input_shape)
    t = 2
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = RRCNN_block(inputs, filters[0], t=t)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = RRCNN_block(e2, filters[1], t=t)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = RRCNN_block(e3, filters[2], t=t)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = RRCNN_block(e4, filters[3], t=t)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = RRCNN_block(e5, filters[4], t=t)

    d5 = up_conv(e5, filters[3])
    d5 = Concatenate()([e4, d5])
    d5 = RRCNN_block(d5, filters[3], t=t)

    d4 = up_conv(d5, filters[2])
    d4 = Concatenate()([e3, d4])
    d4 = RRCNN_block(d4, filters[2], t=t)

    d3 = up_conv(d4, filters[1])
    d3 = Concatenate()([e2, d3])
    d3 = RRCNN_block(d3, filters[1], t=t)

    d2 = up_conv(d3, filters[0])
    d2 = Concatenate()([e1, d2])
    d2 = RRCNN_block(d2, filters[0], t=t)


    x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(d2)

    model = Model(inputs, x)

    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss=['binary_crossentropy'],
                  metrics=[dice_coef])

    return model