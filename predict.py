# this file is the same as test.py only different it tests a set of samples, I will not write detail comments
from __future__ import print_function
import argparse
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from model.multpoolunet import multpoolunet
from dataset_parser.prepareData import VOCPalette
from model.multunet import multires
from model.SEunet import SEunet
from model.unet import unet
from model.fuzzy_unet import fuzzy_unet
from model.FuzzyU import FuzzyU
from model.PSAunet import PSAunet
from model.PSA_eca_unet import PSA_eca_unet
from model.CBAMunet import CBAMunet
from model.multunet_CBAM import multunet_CBAM
from model.ECAunet import ECAunet
from model.attentionunet import attentionunet
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
from model.ResUnet_a import ResUNet_Model
from model.Multpoolunet_1 import Multpoolunet_1
from model.Multpoolunet_2 import Multpoolunet_2
from model.Multpoolunet_3 import Multpoolunet_3
from model.Multpoolunet_4 import Multpoolunet_4
from model.Multpoolunet_5 import Multpoolunet_5
from model.Multpoolunet_6 import Multpoolunet_6
from model.Transunet._model_transunet_2d import transunet_2d
from model.Transunet._model_u2net_2d import u2net_2d
from model.Transunet._model_swin_unet_2d import swin_unet_2d
import cv2
import scipy.io as sio



labelcaption = ['background','tumor','fat','mammary','muscle','bottle','bus','car','cat',
                'chair','cow','Dining table','dog','horse','Motor bike','person','Potted plant',
                'sheep','sofa','train','monitor']

def result_map_to_img(res_map):
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2).astype('uint8')

    return argmax_idx

def findObject(labelimg):
    counts = np.zeros(20,dtype=np.int32)
    str_obj = ''
    for i in range(20):
        counts[i] = np.sum(labelimg == i+1)
        if counts[i] > 500 :
            str_obj = str_obj + labelcaption[i+1]+ ' '
    return str_obj

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", default='Multpoolunet_2', required=False, choices=['multresunet', 'attentionunet', 'ECAunet', 'SEunet', 'multunet_CBAM', 'CBAMunet', 'multunet', 'PSA_eca_unet', 'FuzzyU', 'unet', 'PSAunet', 'pspnet', 'fuzzyunet'],
                    help="Model to train. 'fcn', 'unet', 'PSA_eca_unet', 'PSAunet', 'FuzzyU', 'fuzzyunet'is available.")
parser.add_argument("-P", "--img_path", required=False, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path
vgg_path = None

img_path = './xunlianji/test/img/'
# path of ground truth
label_path = './xunlianji/test/label/'
# path of the txt file contain name list
test_file = './xunlianji/test/test.txt'
# result path one for image
result_path = './result/img/img(Multpoolunet_2)/'

# img_path = './xunlianji/FALLMUD/img3/test/img/'
# # path of ground truth
# label_path = './xunlianji/FALLMUD/img3/test/1/'
# # path of the txt file contain name list
# test_file = './xunlianji/FALLMUD/img3/test/test.txt'
# # result path one for image
# result_path = './result/img_FALLMUD/img(swin_unet)/'

# img_path = './xunlianji/ruxian/test/img/'
# label_path = './xunlianji/ruxian/test/label/'
# test_file = './xunlianji/ruxian/test/test.txt'
# result_path = './result/img_ruxian/img(swin_unet)/'

# img_path = './xunlianji/jiazhuangxian/test/img/'
# label_path = './xunlianji/jiazhuangxian/test/label/'
# test_file = './xunlianji/jiazhuangxian/test/test.txt'
# result_path = './result/img_jiazhuangxian/img(swin_unet)/'

img_width = 256
img_height = 256
nb_class = 2
channels = 3

# Create model to train
print('Creating network...\n')
if model_name == "multpoolunet":
    model = multpoolunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                   lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "PSAunet":
    model = PSAunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "fuzzyunet":
    model = fuzzy_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=0.0001, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "FuzzyU":
    model = FuzzyU(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=0.0001, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "PSA_eca_unet":
    model = PSA_eca_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "multunet":
    model = multires(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                         lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "CBAMunet":
    model = CBAMunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                     lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "multunet_CBAM":
    model = multunet_CBAM(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                     lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "SEunet":
    model = SEunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                          lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "ECAunet":
    model = ECAunet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                   lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "attentionunet":
    model = attentionunet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                          lr_decay=5e-4)
elif model_name == "R2AttUNet":
    model = R2AttUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                      lr_decay=5e-4)
elif model_name == "R2UNet":
    model = R2UNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                   lr_decay=5e-4)
elif model_name == "UnetPlus":
    model = UnetPlus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                     lr_decay=5e-4)
elif model_name == "Unet3Plus":
    model = Unet3Plus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                      lr_decay=5e-4)
elif model_name == "DenseUNet":
    model = DenseUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                      lr_decay=5e-4)
elif model_name == "MultiResUnet":
    model = MultiResUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                         lr_decay=5e-4)
elif model_name == "scSEUnet":
    model = scSEUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                     lr_decay=5e-4)
elif model_name == "DCUNet":
    model = DCUNet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                   lr_decay=5e-4)
elif model_name == "ResUnet":
    model = ResUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                    lr_decay=5e-4)
elif model_name == "DoubleUnet":
    model = DoubleUnet(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                       lr_decay=5e-4)
elif model_name == "ResUnetPlusPlus":
    model = ResUnetPlusPlus(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "transunet_2d":
    model = transunet_2d(input_shape=(img_height, img_width, channels), filter_num=[64, 128, 256, 512],
                         num_classes=nb_class, lr_init=1e-4, lr_decay=5e-4)
elif model_name == "u2net_2d":
    model = u2net_2d(input_shape=(img_height, img_width, channels), filter_num_down=[64, 128, 256, 512],
                     num_classes=nb_class, lr_init=1e-4, lr_decay=5e-4)
elif model_name == "swin_unet_2d":
    model = swin_unet_2d(input_shape=(img_height, img_width, channels), filter_num_begin=64, num_classes=nb_class,
                         lr_init=1e-4, lr_decay=5e-4,
                         depth=4, stack_num_down=2, stack_num_up=2, patch_size=(16, 16), num_heads=[4, 8, 16, 16],
                         window_size=[4, 2, 2, 2], num_mlp=2)
elif model_name == "Multpoolunet_1":
    model = Multpoolunet_1(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "Multpoolunet_2":
    model = Multpoolunet_2(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "Multpoolunet_3":
    model = Multpoolunet_3(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "Multpoolunet_4":
    model = Multpoolunet_4(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "Multpoolunet_5":
    model = Multpoolunet_5(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "Multpoolunet_6":
    model = Multpoolunet_6(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4,
                            lr_decay=5e-4)
elif model_name == "ResUnet_a":
    model = ResUNet_Model().resunet_a_d6(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-4, lr_decay=5e-4)


epochs = 100
try:
    model.load_weights(model_name + '_' + str(epochs) + '_model_weight.h5')
    # model.load_weights(model_name + '_FALLMUD_' + str(epochs) + '_model_weight.h5')
    # model.load_weights(model_name + '_ruxian_' + str(epochs) + '_model_weight.h5')
    # model.load_weights(model_name + '_jiazhuangxian_' + str(epochs) + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

palette = VOCPalette(nb_class=nb_class)

with open(test_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

for i in range(nb_data_img):
    Xpath = img_path + "{}.png".format(namesimg[i])
    Ypath = label_path + "{}.png".format(namesimg[i])

    # read image
    imgorg = Image.open(Xpath)
    imglab = Image.open(Ypath)
    img = imgorg.resize((img_width, img_height), Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = img_arr / 127.5 - 1
    img_arr = np.expand_dims(img_arr, 0)
    #img_arr = img_arr.reshape((1, img_arr.shape[1], img_arr.shape[2], 1))
    # feed the network
    pred = model.predict(img_arr)
    pred_test = pred[0]
    pred_test_res = cv2.resize(pred_test, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
    res = result_map_to_img(pred[0])
    # save the probability as .mat
    # dataNew = result_path2+"{}.mat".format(namesimg[i])
    # sio.savemat(dataNew, {'A': pred_test_res})
    PIL_img_pal = palette.genlabelpal(res)
    plt.imshow(PIL_img_pal)
    PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)
    b = "{}.png".format(namesimg[i])
    PIL_img_pal.save(result_path + b)




