#from __future__ import annotations
#from __future__ import print_function
# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import argparse
from keras.callbacks import ModelCheckpoint
from callbacks import TrainCheck
from model.multunet import multires
from model.unet import unet
from model.fuzzy_unet import fuzzy_unet
from model.PSAunet import PSAunet
from model.multpoolunet import multpoolunet
from model.PSA_eca_unet import PSA_eca_unet
from model.CBAMunet import CBAMunet
from model.ECAunet import ECAunet
from model.multunet_CBAM import multunet_CBAM
# from model._model_r2_unet import att_unet, r2_unet, att_r2_unet
from model.fcn import fcn_8s
from model.attentionunet import attentionunet
from model.SEunet import SEunet
from model.R2AttUnet import R2AttUNet
from model.R2UNet import R2UNet
from model.UnetPlus import UnetPlus
from model.UNet3Plus import Unet3Plus
from model.DenseUnet import DenseUNet
from model.DCunet import DCUNet
from model.MultiResUnet import MultiResUnet
from model.SCSEunet import scSEUnet
from model.ResUnet import ResUnet
from model.DoubleUnet import DoubleUnet
from model.ResuentPlusPlus import ResUnetPlusPlus
from model.Transunet._model_transunet_2d import transunet_2d
from model.Transunet._model_u2net_2d import u2net_2d
from model.Transunet._model_swin_unet_2d import swin_unet_2d
from model.Multpoolunet_1 import Multpoolunet_1
from model.Multpoolunet_2 import Multpoolunet_2
from model.Multpoolunet_3 import Multpoolunet_3
from model.Multpoolunet_4 import Multpoolunet_4
from model.Multpoolunet_5 import Multpoolunet_5
from model.Multpoolunet_6 import Multpoolunet_6
from dataset_parser.generator import data_generator_dir
from model.ResUnet_a import ResUNet_Model
# from caculate_params import stats_graph
import tensorflow as tf
import os
import time

# gpus = tf.config.list_physical_devices("GPU")
#
# if gpus:
#     gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
#     # tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
#     # 或者也可以设置GPU显存为固定使用量(例如：4G)
#     tf.config.experimental.set_virtual_device_configuration(gpu0,
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    # tf.config.set_visible_devices([gpu0], "GPU")

# config = tf.compat.v1.ConfigProto()
# #设置最大占有GPU不超过显存的70%
# config.gpu_options.per_process_gpu_memory_fraction=0.3
# #重点：设置动态分配GPU
# config.gpu_options.allow_growth=True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# # Parse Options
parser = argparse.ArgumentParser()
# parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet', 'fuzzyunet', 'SCFnet'],
#                    help="Model to train. 'fcn', 'unet', 'pspnet', 'fuzzyunet', 'SCFnet'is available.")
# parser.add_argument("-M", "--model", default="FuzzyU", required=False, choices=['fcn', 'FuzzyU', 'unet', 'pspnet', 'fuzzyunet', 'SCFnet'],
#                     help="Model to train. 'fcn', 'unet',FuzzyU , 'pspnet', 'fuzzyunet', 'SCFnet'is available.")
parser.add_argument("-M", "--model", default="Multpoolunet_2", required=False, choices=['R2AttUNet', 'r2_unet', 'transunet_2d', 'multpoolunet', 'attentionunet', 'SEunet', 'ECAunet', 'multunet_CBAM', 'CBAMunet', 'multresunet', 'PSA_eca_unet','fuzzyunet', 'multunet'],
                     help="Model to train. 'fcn', 'PSAunet','unet','FuzzyU', 'PSAunet', 'pspnet', 'fuzzyunet', 'SCFnet'is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=4, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg

# epoch
epochs = 100
# continue training
resume_training = False

# path of document
# load the txt files contain names of images in train set
path_to_train = './xunlianji/train.txt'
path_to_val = './xunlianji/val.txt'
path_to_img = './xunlianji/img/'
path_to_label = './xunlianji/label/'

# path_to_train = './xunlianji/FALLMUD/img3/train/train.txt'
# path_to_val = './xunlianji/FALLMUD/img3/train/val.txt'
# path_to_img = './xunlianji/FALLMUD/img3/train/img/'
# path_to_label = './xunlianji/FALLMUD/img3/train/1/'

# path_to_train = './xunlianji/ruxian/train/train.txt'
# path_to_val = './xunlianji/ruxian/train/val.txt'
# path_to_img = './xunlianji/ruxian/train/img/'
# path_to_label = './xunlianji/ruxian/train/label/'

# path_to_train = './xunlianji/jiazhuangxian/train/train.txt'
# path_to_val = './xunlianji/jiazhuangxian/train/val.txt'
# path_to_img = './xunlianji/jiazhuangxian/train/img/'
# path_to_label = './xunlianji/jiazhuangxian/train/label/'

# the size of input layer
# image size
img_width = 256
img_height = 256

# category number
nb_class = 2

# input image channel
channels = 3

# read the name in train set
with open(path_to_train,"r") as f:
    ls = f.readlines()
namestrain = [l.rstrip('\n') for l in ls]
nb_data_train = len(namestrain)
# read the name in validation set
with open(path_to_val,"r") as f:
    ls = f.readlines()
namesval = [l.rstrip('\n') for l in ls]
nb_data_val = len(namesval)

# Create model to train
print('Creating network...\n')
if model_name == "unet":
     model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                  lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "fuzzyunet":
    model = fuzzy_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
               lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "PSAunet":
    model = PSAunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
               lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "multpoolunet":
    model = multpoolunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
               lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "multunet":
    model = multires(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "PSA_eca_unet":
    model = PSA_eca_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "CBAMunet":
    model = CBAMunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "multunet_CBAM":
    model = multunet_CBAM(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "ECAunet":
    model = ECAunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "SEunet":
    model = SEunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "attentionunet":
    model = attentionunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "fcn":
    model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "R2AttUNet":
    model = R2AttUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "R2UNet":
    model = R2UNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "UnetPlus":
    model = UnetPlus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Unet3Plus":
    model = Unet3Plus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "DenseUNet":
    model = DenseUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "MultiResUnet":
    model = MultiResUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "scSEUnet":
    model = scSEUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "DCUNet":
    model = DCUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "ResUnet":
    model = ResUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "DoubleUnet":
    model = DoubleUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "ResUnetPlusPlus":
    model = ResUnetPlusPlus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "transunet_2d":
    model = transunet_2d(input_shape=(img_height, img_width, channels), filter_num=[64, 128, 256, 512], num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "u2net_2d":
    model = u2net_2d(input_shape=(img_height, img_width, channels), filter_num_down=[64, 128, 256, 512], num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "swin_unet_2d":
    model = swin_unet_2d(input_shape=(img_height, img_width, channels), filter_num_begin=64, num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay,
                         depth=4, stack_num_down=2, stack_num_up=2, patch_size=(16, 16), num_heads=[4, 8, 16, 16], window_size=[4, 2, 2, 2], num_mlp=2)
elif model_name == "Multpoolunet_1":
    model = Multpoolunet_1(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Multpoolunet_2":
    model = Multpoolunet_2(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Multpoolunet_3":
    model = Multpoolunet_3(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Multpoolunet_4":
    model = Multpoolunet_4(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Multpoolunet_5":
    model = Multpoolunet_5(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "Multpoolunet_6":
    model = Multpoolunet_6(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "ResUnet_a":
    model = ResUNet_Model().resunet_a_d6(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)

# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_' + str(epochs) + '_model_checkpoint_weight.h5',
                             # filepath=model_name + '_FALLMUD_' + str(epochs) + '_model_checkpoint_weight.h5',
                             # filepath=model_name + '_ruxian_' + str(epochs) + '_model_checkpoint_weight.h5',
                             # filepath=model_name + '_jiazhuangxian_' + str(epochs) + '_model_checkpoint_weight.h5',
                             monitor='val_dice_coef',
                             # monitor='val_categorical_accuracy',
                             save_best_only=True,
                             save_weights_only=False)
train_check = TrainCheck(output_path='./img', model_name=model_name,
                         img_shape=(img_height, img_width), nb_class=nb_class)

# load weights if needed
if resume_training:
    print('Resume training...\n')
    model.load_weights(model_name + '_' + str(epochs) + '_model_checkpoint_weight.h5')
    # model.load_weights(model_name + '_FALLMUD_' + str(epochs) + '_model_checkpoint_weight.h5')
    # model.load_weights(model_name + '_ruxian_' + str(epochs) + '_model_checkpoint_weight.h5')
    # model.load_weights(model_name + '_jiazhuangxian_' + str(epochs) + '_model_checkpoint_weight.h5')

else:
    print('New training...\n')

# training
history = model.fit_generator(data_generator_dir(namestrain, path_to_img, path_to_label, (img_height, img_width, channels), nb_class, TRAIN_BATCH, 'train'),
                              steps_per_epoch=nb_data_train // TRAIN_BATCH,
                              validation_data=data_generator_dir(namesval, path_to_img, path_to_label, (img_height, img_width, channels), nb_class, VAL_BATCH, 'val'),
                              validation_steps=nb_data_val // VAL_BATCH,
                              callbacks=[checkpoint, train_check],
                              epochs=epochs,
                              verbose=1)


# serialize model weigths to h5
model.save_weights(model_name + '_' + str(epochs) + '_model_weight.h5')
# model.save_weights(model_name + '_FALLMUD_' + str(epochs) + '_model_weight.h5')
# model.save_weights(model_name + '_ruxian_' + str(epochs) + '_model_weight.h5')
# model.save_weights(model_name + '_jiazhuangxian_' + str(epochs) + '_model_weight.h5')


# save the training loss and validation loss to txt
f_loss = open(model_name + '_' + str(epochs)  + "loss.txt", "a")
# f_loss = open(model_name + '_FALLMUD_' + str(epochs) + "loss.txt", "a")
# f_loss = open(model_name + '_ruxian_' + str(epochs) + "loss.txt", "a")
# f_loss = open(model_name + '_jiazhuangxian_' + str(epochs) + "loss.txt", "a")

# f_loss.write(str(history.history))
# f_loss.close()
