from keras import Input
from keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add, \
    BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.optimizers import adam_v2

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def AttnBlock2D(x, g, inter_channel):
    # x: skip connection layer
    # g: down layer upsampling 后的 layer
    # inner_channel: down layer 的通道数 // 4

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x


def attention_up_and_concate(down_layer, layer):
    # down_layer: 承接下来的 layer
    # layer: skip connection layer

    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = AttnBlock2D(x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])

    return concate


# Attention U-Net
def attentionunet(num_classes, input_shape, lr_init, lr_decay):
    inputs = Input(input_shape)
    x = inputs
    depth = 4
    features = 64
    skips = []

    # depth = 0, 1, 2, 3
    # ENCODER
    for i in range(depth):
        x = Conv2D(features, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    # BOTTLENECK
    x = Conv2D(features, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(features, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # DECODER
    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i])
        x = Conv2D(features, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model

