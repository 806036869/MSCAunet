from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras import backend as K


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def conv_block(input_tensor, num_of_channels, kernel_size=3):
    x = Conv2D(num_of_channels, (kernel_size, kernel_size), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_of_channels, (kernel_size, kernel_size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def UnetPlus(num_classes, input_shape, lr_init, lr_decay):
    # Build and train our neural network
    inputs = Input(input_shape)

    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(64, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1)


    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(128, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2)

    up1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], axis=3)
    conv1_2 = conv_block(conv1_2, num_of_channels=64)

    c3 = Conv2D(256, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(256, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D()(c3)

    up2_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    conv2_2 = concatenate([up2_2, c2], axis=3)
    conv2_2 = conv_block(conv2_2, num_of_channels=128)

    up1_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, c1, conv1_2], axis=3)
    conv1_3 = conv_block(conv1_3, num_of_channels=64)

    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D()(c4)

    up3_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    conv3_2 = concatenate([up3_2, c3], axis=3)
    conv3_2 = conv_block(conv3_2, num_of_channels=256)

    up2_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], axis=3)
    conv2_3 = conv_block(conv2_3, num_of_channels=128)

    up1_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, c1, conv1_2, conv1_3], axis=3)
    conv1_4 = conv_block(conv1_4, num_of_channels=64)

    c5 = Conv2D(1024, (3, 3), padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(1024, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    up4_2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    conv4_2 = concatenate([up4_2, c4], axis=3)
    conv4_2 = conv_block(conv4_2, num_of_channels=512)

    up3_3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, c3, conv3_2], axis=3)
    conv3_3 = conv_block(conv3_3, num_of_channels=256)

    up2_4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], axis=3)
    conv2_4 = conv_block(conv2_4, num_of_channels=128)

    up1_5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, c1, conv1_2, conv1_3, conv1_4], axis=3)
    conv1_5 = conv_block(conv1_5, num_of_channels=64)

    x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(conv1_5)

    model = Model(inputs, x)
    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss=['binary_crossentropy'],
                  metrics=[dice_coef])
                  # metrics='binary_accuracy')
    return model