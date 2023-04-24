from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, UpSampling2D, ReLU, Add, \
    Lambda
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

# class UNet(object):
#     def __init__(self, config=UnetConfig()):
    #     self.config = config
    #     self.model = self.build_model_resUnet()
    #
    # def build_model_resUnet(self):
    #     def Tanimoto_loss(label, pred):
    #         square = square(pred)
    #         sum_square = reduce_sum(square, axis=-1)
    #         product = multiply(pred, label)
    #         sum_product = reduce_sum(product, axis=-1)
    #         denomintor = subtract(add(sum_square, 1), sum_product)
    #         loss = divide(sum_product, denomintor)
    #         loss = reduce_mean(loss)
    #         return 1.0 - loss
    #
    #     def Tanimoto_dual_loss(label, pred):
    #         loss1 = Tanimoto_loss(pred, label)
    #         pred = subtract(1.0, pred)
    #         label = subtract(1.0, label)
    #         loss2 = Tanimoto_loss(label, pred)
    #         loss = (loss1 + loss2) / 2
    #
    #     def ResBlock(input, filter, kernel_size, dilation_rates, stride):
    #         def branch(dilation_rate):
    #             x = KL.BatchNormalization()(input)
    #             x = KL.Activation('relu')(x)
    #             x = KL.Conv2D(filter, kernel_size, strides=stride, dilation_rate=dilation_rate, padding='same')(x)
    #             x = KL.BatchNormalization()(x)
    #             x = KL.Activation('relu')(x)
    #             x = KL.Conv2D(filter, kernel_size, strides=stride, dilation_rate=dilation_rate, padding='same')(x)
    #             return x
    #
    #         out = []
    #         for d in dilation_rates:
    #             out.append(branch(d))
    #         if len(dilation_rates) > 1:
    #             out = KL.Add()(out)
    #         else:
    #             out = out[0]
    #         return out
    #
    #     def PSPPooling(input, filter):
    #         x1 = KL.MaxPooling2D(pool_size=(2, 2))(input)
    #         x2 = KL.MaxPooling2D(pool_size=(4, 4))(input)
    #         x3 = KL.MaxPooling2D(pool_size=(8, 8))(input)
    #         x4 = KL.MaxPooling2D(pool_size=(16, 16))(input)
    #         x1 = KL.Conv2D(int(filter / 4), (1, 1))(x1)
    #         x2 = KL.Conv2D(int(filter / 4), (1, 1))(x2)
    #         x3 = KL.Conv2D(int(filter / 4), (1, 1))(x3)
    #         x4 = KL.Conv2D(int(filter / 4), (1, 1))(x4)
    #         x1 = KL.UpSampling2D(size=(2, 2))(x1)
    #         x2 = KL.UpSampling2D(size=(4, 4))(x2)
    #         x3 = KL.UpSampling2D(size=(8, 8))(x3)
    #         x4 = KL.UpSampling2D(size=(16, 16))(x4)
    #         x = KL.concatenate()([x1, x2, x3, x4, input])
    #         x = KL.Conv2D(filter, (1, 1))(x)
    #         return x
    #
    #     def combine(input1, input2, filter):
    #         x = KL.Activation('relu')(input1)
    #         x = KL.concatenate()([x, input2])
    #         x = KL.Conv2D(filter, (1, 1))(x)
    #         return x
    #
    #     inputs = KM.Input(shape=(self.config.IMAGE_H, self.config.IMAGE_W, self.config.IMAGE_C))
    #     c1 = x = KL.Conv2D(32, (1, 1), strides=(1, 1), dilation_rate=1)(inputs)
    #     c2 = x = ResBlock(x, 32, (3, 3), [1, 3, 15, 31], (1, 1))
    #     x = KL.Conv2D(64, (1, 1), strides=(2, 2))(x)
    #     c3 = x = ResBlock(x, 64, (3, 3), [1, 3, 15, 31], (1, 1))
    #     x = KL.Conv2D(128, (1, 1), strides=(2, 2))(x)
    #     c4 = x = ResBlock(x, 128, (3, 3), [1, 3, 15], (1, 1))
    #     x = KL.Conv2D(256, (1, 1), strides=(2, 2))(x)
    #     c5 = x = ResBlock(x, 256, (3, 3), [1, 3, 15], (1, 1))
    #     x = KL.Conv2D(512, (1, 1), strides=(2, 2))(x)
    #     c6 = x = ResBlock(x, 512, (3, 3), [1], (1, 1))
    #     x = KL.Conv2D(1024, (1, 1), strides=(2, 2))(x)
    #     x = ResBlock(x, 1024, (3, 3), [1], (1, 1))
    #     x = PSPPooling(x, 1024)
    #     x = KL.Conv2D(512, (1, 1))(x)
    #     x = KL.UpSampling2D()(x)
    #     x = combine(x, c6, 512)
    #     x = ResBlock(x, 512, (3, 3), [1], 1)
    #     x = KL.Conv2D(256, (1, 1))(x)
    #     x = KL.UpSampling2D()(x)
    #     x = combine(x, c5, 256)
    #     x = ResBlock(x, 256, (3, 3), [1, 3, 15], 1)
    #     x = KL.Conv2D(128, (1, 1))(x)
    #     x = KL.UpSampling2D()(x)
    #     x = combine(x, c4, 128)
    #     x = ResBlock(x, 128, (3, 3), [1, 3, 15], 1)
    #     x = KL.Conv2D(64, (1, 1))(x)
    #     x = KL.UpSampling2D()(x)
    #     x = combine(x, c3, 64)
    #     x = ResBlock(x, 64, (3, 3), [1, 3, 15, 31], 1)
    #     x = KL.Conv2D(32, (1, 1))(x)
    #     x = KL.UpSampling2D()(x)
    #     x = combine(x, c2, 32)
    #     x = ResBlock(x, 32, (3, 3), [1, 3, 15, 31], 1)
    #     x = combine(x, c1, 32)
    #     x = PSPPooling(x, 32)
    #     x = KL.Conv2D(self.config.CLASSES_NUM, (1, 1))(x)
    #     x = KL.Activation('softmax')(x)
    #     model = KM.Model(inputs=inputs, outputs=x)
    #     model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.8), loss=Tanimoto_loss, metrics=['accuracy'])
    #     model.summary()
    #     return model
class ResUNet_Model:
    def __res_block(self, x, filters, kernel_size, strides, dilation_rate):
        def arm(dilation):
            batch_norm = BatchNormalization()(x)
            relu = ReLU()(batch_norm)
            conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                          dilation_rate=dilation, padding='same')(relu)
            batch_norm = BatchNormalization()(conv)
            relu = ReLU()(batch_norm)
            conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                          dilation_rate=dilation, padding='same')(relu)
            return conv

        outcomes = [x]  # might cause problems with size...
        for dilation in dilation_rate:
            outcomes.append(arm(dilation))

        value = (Add()(outcomes))

        return value

    def __psp_pooling(self, x, features):
        def unit(index, pool_size):
            # split= split into 1/4
            split = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 4})(x)[index]
            # split = tf.split(value=x, num_or_size_splits=4, axis=3)[
            #     index]  # axis=0 is the default, but not sure is correct... need dyn. index?

            max_pool = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='same')(split)

            restore_dim = UpSampling2D(size=pool_size, interpolation='bilinear')(max_pool)

            conv2dbn = self.__conv2d_bn(restore_dim, filters=(features // 4), kernel_size=(1, 1), strides=(1, 1),
                                        dilation_rate=1)

            return conv2dbn

        pool_size = [(1, 1), (2, 2), (4, 4), (8, 8)]
        outcomes = [x]
        i = 1
        while i <= 4:
            outcomes.append(unit((i - 1), pool_size=pool_size[i - 1]))
            i += 1

        concat = concatenate(outcomes)
        conv2dbn = self.__conv2d_bn(concat, filters=features, kernel_size=(1, 1), strides=(1, 1), dilation_rate=1)
        return conv2dbn

    def __conv2d_bn(self, x, filters, kernel_size, strides, dilation_rate):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                      dilation_rate=dilation_rate)(x)
        batch_norm = BatchNormalization()(conv)
        return batch_norm

    def __combine(self, filters, x1, x2):
        relu = ReLU()(x1)
        concat = concatenate([relu, x2])
        conv2dbn = self.__conv2d_bn(concat, filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=1)
        return conv2dbn

    def __upsample(self, x, filters):
        # double input (see figure 1 of paper)
        upsample = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        conv2dbn = self.__conv2d_bn(upsample, filters=filters, kernel_size=(1, 1), strides=(1, 1),
                                    dilation_rate=1)  # not sure about strides here...
        return conv2dbn

    def resunet_a_d6(self, input_shape, num_classes, lr_init, lr_decay):
        inputs = Input(input_shape)
        # Encoder
        enc1 = self.__conv2d_bn(inputs, filters=64, kernel_size=(1, 1), strides=(1, 1), dilation_rate=1)
        enc2 = self.__res_block(enc1, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1, 3, 15, 31])
        enc3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2))(enc2)
        enc4 = self.__res_block(enc3, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1, 3, 15, 31])
        enc5 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2))(enc4)
        enc6 = self.__res_block(enc5, filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1, 3, 15])
        enc7 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2))(enc6)
        enc8 = self.__res_block(enc7, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1, 3, 15])
        enc9 = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2))(enc8)
        enc10 = self.__res_block(enc9, filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        # enc11 = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2))(enc10)
        # enc12 = self.__res_block(enc11, filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        # Pooling bridge
        psp = self.__psp_pooling(enc10, features=1024)
        # Decoder
        dec14 = self.__upsample(psp, filters=512)
        dec15 = self.__combine(filters=512, x1=dec14, x2=enc8)
        dec16 = self.__res_block(dec15, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        dec17 = self.__upsample(dec16, filters=256)
        dec18 = self.__combine(filters=256, x1=dec17, x2=enc6)
        dec19 = self.__res_block(dec18, filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        dec20 = self.__upsample(dec19, filters=128)
        dec21 = self.__combine(filters=128, x1=dec20, x2=enc4)
        dec22 = self.__res_block(dec21, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        dec23 = self.__upsample(dec22, filters=64)
        dec24 = self.__combine(filters=64, x1=dec23, x2=enc2)
        dec25 = self.__res_block(dec24, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        # dec26 = self.__upsample(dec25, filters=32)
        # dec27 = self.__combine(filters=32, x1=dec26, x2=enc2)
        # dec28 = self.__res_block(dec27, filters=32, kernel_size=(3, 3), strides=(1, 1), dilation_rate=[1])
        dec29 = self.__combine(filters=64, x1=dec25, x2=enc1)
        # PSP Pooling
        psp = self.__psp_pooling(dec29, features=64)
        x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(psp)

        # changing from axis=1 to axis=-1 fixed this...
        # return model
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=adam_v2.Adam(lr=lr_init, decay=lr_decay),
                      loss='binary_crossentropy',
                      metrics=[dice_coef])
        return model