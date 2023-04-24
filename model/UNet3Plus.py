from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, UpSampling2D, average
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras import backend as K
from keras.models import Model
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

# [pytorch-Unet3](https://github.com/ZJUGiveLab/UNet-Version),[paper](https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
def conv_block(inputs, filters, kernel_size=3, strides=1, padding='same'):
    Z = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    Z = BatchNormalization()(Z)
    A = Activation('relu')(Z)
    return A

def UnetConv2(inputs,filters,n=2,kernel_size=3,stride=1,padding='same'):
    x=inputs
    for i in range(0,n+1):
        x = Conv2D(filters,kernel_size=kernel_size,strides=(stride,stride),padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def Unet3Plus(num_classes, input_shape, lr_init, lr_decay,n=2,filters = [64, 128, 256, 512, 1024] ):
    #filters = [64, 128, 256, 512, 1024]
    #filters = [32, 64, 128, 256, 512] ,n=1 23,868,805
    CatChannels = filters[0]
    CatBlocks = 5
    UpChannels = CatChannels * CatBlocks
    inputs=Input(shape=input_shape)
    h1 = UnetConv2(inputs, filters[0],n=n)

    h2 = MaxPooling2D(strides=(2,2))(h1)
    h2 = UnetConv2(h2, filters[1],n=n) #shape=(None, 160, 160, 128)

    h3 = MaxPooling2D(strides=(2, 2))(h2)
    h3 = UnetConv2(h3, filters[2],n=n) #shape=(None, 80, 80, 256)

    h4 = MaxPooling2D(strides=(2, 2))(h3)
    h4 = UnetConv2(h4, filters[3],n=1) #shape=(None, 40, 40, 512)

    h5 = MaxPooling2D(strides=(2, 2))(h4)
    hd5 = UnetConv2(h5, filters[4],n=n) #shape=(None, 20, 20, 1024)

    h1_PT_hd4 = MaxPooling2D(strides=(8, 8))(h1)
    h1_PT_hd4 = conv_block(h1_PT_hd4, filters[0]) #shape=(None, 40, 40, 64

    h2_PT_hd4 = MaxPooling2D(strides=(4, 4))(h2)
    h2_PT_hd4 = conv_block(h2_PT_hd4, filters[1]) #shape=(None, 40, 40, 128)

    h3_PT_hd4 = MaxPooling2D(strides=(2, 2))(h3)
    h3_PT_hd4 = conv_block(h3_PT_hd4, filters[2]) # shape=(None, 40, 40, 256)

    h4_Cat_hd4=conv_block(h4, filters[3]) #shape=(None, 40, 40, 512)

    hd5_UT_hd4= UpSampling2D(size=(2,2))(hd5)
    hd5_UT_hd4=conv_block(hd5_UT_hd4,filters[4]) #shape=(None, 40, 40, 1024)

    #fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
    hd4 = concatenate([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4]) #shape=(None, 40, 40, 1984)
    hd4 = conv_block(hd4, UpChannels) #shape=(None, 40, 40, 320)

    #stage 3d
    h1_PT_hd3 = MaxPooling2D(strides=(4, 4))(h1)
    h1_PT_hd3=conv_block(h1_PT_hd3, filters[0]) #shape=(None, 80, 80, 64)

    h2_PT_hd3 = MaxPooling2D(strides=(2, 2))(h2)
    h2_PT_hd3 = conv_block(h2_PT_hd3, filters[1]) #shape=(None, 80, 80, 128)

    h3_Cat_hd3=conv_block(h3, filters[2]) #shape=(None, 80, 80, 256)

    hd4_UT_hd3 = UpSampling2D(size=(2,2))(hd4)
    hd4_UT_hd3 =conv_block(hd4_UT_hd3,UpChannels) #shape=(None, 80, 80, 320)

    hd5_UT_hd3=UpSampling2D(size=(4,4))(hd5)
    hd5_UT_hd3 = conv_block(hd5_UT_hd3, UpChannels) #shape=(None, 80, 80, 320)

    # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
    hd3=concatenate([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3])
    hd3 = conv_block(hd3, UpChannels) #shape=(None, 80, 80, 320)

    #stage 2d
    h1_PT_hd2 = MaxPooling2D(strides=(2, 2))(h1)
    h1_PT_hd2 = conv_block(h1_PT_hd2, filters[0]) #shape=(None, 160, 160, 64)

    h2_Cat_hd2=conv_block(h2, filters[1]) #shape=(None, 160, 160, 128)

    hd3_UT_hd2=UpSampling2D(size=(2,2))(hd3)
    hd3_UT_hd2 = conv_block(hd3_UT_hd2, UpChannels) #shape=(None, 160, 160, 320)

    hd4_UT_hd2 = UpSampling2D(size=(4,4))(hd4)
    hd4_UT_hd2 = conv_block(hd4_UT_hd2, UpChannels) # shape=(None, 160, 160, 320)

    hd5_UT_hd2 = UpSampling2D(size=(8, 8))(hd5)
    hd5_UT_hd2 = conv_block(hd5_UT_hd2, UpChannels)  # shape=(None, 160, 160, 320)

    ## fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
    hd2=concatenate([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2])
    hd2 = conv_block(hd2, UpChannels) #shape=(None, 160, 160, 320)

    #stage 1d
    h1_Cat_hd1 = conv_block(h1, filters[0]) # shape=(None, 320, 320, 64)

    hd2_UT_hd1 =UpSampling2D(size=(2,2))(hd2)
    hd2_UT_hd1 = conv_block(hd2_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd3_UT_hd1=UpSampling2D(size=(4,4))(hd3)
    hd3_UT_hd1 = conv_block(hd3_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd4_UT_hd1=UpSampling2D(size=(8,8))(hd4)
    hd4_UT_hd1 = conv_block(hd4_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd5_UT_hd1 = UpSampling2D(size=(16, 16))(hd5)
    hd5_UT_hd1 = conv_block(hd5_UT_hd1, UpChannels)  # shape=(None, 320, 320, 320)

    #fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
    hd1 = concatenate([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1])
    hd1 = conv_block(hd1, UpChannels)  # shape=(None, 320, 320, 320)

    d5 = Conv2D(num_classes,kernel_size=3,activation=None,padding='same',use_bias=False)(hd5)
    d5 = UpSampling2D(size=(16, 16))(d5)
    d55=Activation('sigmoid',name='d5')(d5)

    d4 =Conv2D(num_classes,kernel_size=3,activation=None,padding='same',use_bias=False)(hd4)
    d4 = UpSampling2D(size=(8, 8)  )(d4)
    d44 = Activation('sigmoid',name='d4')(d4)

    d3 = Conv2D(num_classes,kernel_size=3,activation=None,padding='same',use_bias=False)(hd3)
    d3 = UpSampling2D(size=(4, 4)  )(d3)
    d33 = Activation('sigmoid',name='d3')(d3)

    d2 = Conv2D(num_classes,kernel_size=3,activation=None,padding='same',use_bias=False)(hd2)
    d2 = UpSampling2D(size=(2, 2))(d2)
    d22 = Activation('sigmoid',name='d2')(d2)

    d1 = Conv2D(num_classes,kernel_size=3,activation=None,padding='same',use_bias=False)(hd1)
    #
    d11=Activation('sigmoid',name='d1')(d1)
    d = average([d1, d2, d3, d4, d5])
    d = Conv2D(num_classes, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = Activation('sigmoid', name='d')(d)
    # model = Model(inputs=inputs,outputs=[d,d11,d22,d33,d44,d55])
    model = Model(inputs=inputs, outputs=[d11])
    model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                  loss=['binary_crossentropy'],
                  metrics=[dice_coef])
    return model