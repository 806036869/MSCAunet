from keras.models import Model
from keras.layers import Input, multiply, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, AvgPool2D, Add
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras import backend as K
from model.EPSA_Module import PSAModule, Mluti_Scale_block, ECAWeightModule
import tensorflow as tf

# compute the dice-coefficient 其中smooth目的：防止分母为0
# 只有全局特征模块

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_loss(y_true, y_pred):
    return 1-(2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def Combine_Bce_Dice_Loss(y_true, y_pred):
    bce = tf.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.7*bce+0.3*dice

# network structure
def Multpoolunet_4(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    # input layer
    # ============================================================================
    img_input = Input(input_shape)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    productlayer1 = Activation('relu')(x)

    x = MaxPooling2D()(productlayer1)
    x1 = AvgPool2D(pool_size=(8, 8))(x)
    x12 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1))(x)
    x12 = Conv2D(16, kernel_size=(9, 9),  strides=(8, 8), padding='same')(x12)
    x12 = BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x12 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x12)
    x1 = Add()([x1, x12])

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    productlayer2 = Activation('relu')(x)

    x = MaxPooling2D()(productlayer2)
    x2 = AvgPool2D(pool_size=(4, 4))(x)
    x2 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x2)
    x22 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1))(x)
    x22 = Conv2D(16, kernel_size=(5, 5),  strides=(4, 4), padding='same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = Activation('relu')(x22)
    x22 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x22)
    x2 = Add()([x2, x22])

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    productlayer3 = Activation('relu')(x)

    x = MaxPooling2D()(productlayer3)
    x3 = AvgPool2D(pool_size=(2, 2))(x)
    x3 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x3)
    x33 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1))(x)
    x33 = Conv2D(16, kernel_size=(3, 3),  strides=(2, 2), padding='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)
    x33 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x33)
    x3 = Add()([x3, x33])

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    productlayer4 = Activation('relu')(x)
    x = MaxPooling2D()(productlayer4)

    # Block 5
    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([x, x1, x2, x3])


    for_pretrained_weight = x

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, productlayer4])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, productlayer3])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, productlayer2])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, productlayer1])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss='binary_crossentropy',
                  metrics=[dice_coef])
    return model


