"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import os

import cv2
import numpy as np
import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
__all__ = ['SegmentationMetric']

from matplotlib import pyplot as plt

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        # 准确率 acc
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # 精确率 precision
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # def Specificity(self):
    #     # Specificity = (TN) / TN + FP
    #
    #     return np.diag(self.confusionMatrix) / (np.sum(self.confusionMatrix) - np.sum(self.confusionMatrix, axis=0))

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        # mIoU = [ TP / (TP + FP + FN) + TN / (TN + FN + FP) ] / 2
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    # def Dice_coefficient(self):
    #     # Intersection = TP Union = 2TP + FP + FN
    #     # Dice = 2TP/(2TP+FP+FN)
    #     intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
    #     union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
    #     Dice = 2 * intersection / union  # 返回列表，其值为各个类别的IoU
    #     return Dice


    def recall(self):
        # 召回率 recall= TP/(TP+FN)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        np.seterr(divide='ignore', invalid='ignore')
        return recall

    def precision(self):
        # precosion= TP/(TP+FP)
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return precision

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        # return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    # 计算DICE系数，即DSI
    def calDSI(binary_GT, binary_R):
        row, col = binary_GT.shape  # 矩阵的行与列
        DSI_s, DSI_t = 0, 0
        for i in range(row):
            for j in range(col):
                if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                    DSI_s += 1
                if binary_GT[i][j] == 255:
                    DSI_t += 1
                if binary_R[i][j] == 255:
                    DSI_t += 1
        DSI = 2 * DSI_s / DSI_t
        # print(DSI)
        return DSI

    # 计算VOE系数，即VOE
    # 体素重叠误差 Volumetric Overlap Error
    def calVOE(binary_GT, binary_R):
        row, col = binary_GT.shape  # 矩阵的行与列
        VOE_s, VOE_t = 0, 0
        for i in range(row):
            for j in range(col):
                if binary_GT[i][j] == 255:
                    VOE_s += 1
                if binary_R[i][j] == 255:
                    VOE_t += 1
        VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
        return VOE

    # 计算RVD系数，即RVD
    # （体素相对误差 Relative Volume Difference，也称为VD）
    def calRVD(binary_GT, binary_R):
        row, col = binary_GT.shape  # 矩阵的行与列
        RVD_s, RVD_t = 0, 0
        for i in range(row):
            for j in range(col):
                if binary_GT[i][j] == 255:
                    RVD_s += 1
                if binary_R[i][j] == 255:
                    RVD_t += 1
        RVD = RVD_t / RVD_s - 1
        return RVD

    # 计算Prevision系数，即Precison
    # # 准确度是检测出的像素中是边缘的比例
    # def calPrecision(binary_GT, binary_R):
    #     row, col = binary_GT.shape  # 矩阵的行与列
    #     P_s, P_t = 0, 0
    #     for i in range(row):
    #         for j in range(col):
    #             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
    #                 P_s += 1
    #             if binary_R[i][j] == 255:
    #                 P_t += 1
    #
    #     Precision = P_s / P_t
    #     return Precision

    # 计算Recall系数，即Recall
    # 召回率是检测出的正确边缘占所有边缘的比例
    # def calRecall(binary_GT, binary_R):
    #     row, col = binary_GT.shape  # 矩阵的行与列
    #     R_s, R_t = 0, 0
    #     for i in range(row):
    #         for j in range(col):
    #             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
    #                 R_s += 1
    #             if binary_GT[i][j] == 255:
    #                 R_t += 1
    #
    #     Recall = R_s / R_t
    #     return Recall


# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
# 体素重叠误差 Volumetric Overlap Error
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
# （体素相对误差 Relative Volume Difference，也称为VD）
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
# 准确度是检测出的像素中是边缘的比例
# def calPrecision(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     P_s, P_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
#                 P_s += 1
#             if binary_R[i][j] == 255:
#                 P_t += 1
#
#     Precision = P_s / P_t
#     return Precision


# 计算Recall系数，即Recall
# 召回率是检测出的正确边缘占所有边缘的比例
# def calRecall(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     R_s, R_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
#                 R_s += 1
#             if binary_GT[i][j] == 255:
#                 R_t += 1
#
#     Recall = R_s / R_t
#     return Recall



if __name__ == '__main__':

    pa = []
    cpa = []
    mpa = []
    ioU = []
    meanIoU = []
    dice = []
    recall = []
    F1_score = []
    FWIOU = []
    specificity = []
    precision = []

    DSI = []
    VOE = []
    RVD = []
    Precision = []
    Recall = []

    label_path = './xunlianji/test/label/'
    predict_path = './result/img/img(Multpoolunet_2)/'

    # label_path = './xunlianji/FALLMUD/img3/test/1/'
    # predict_path = './result/img_FALLMUD/img(swin_unet)/'

    # label_path = './xunlianji/ruxian/test/label/'
    # predict_path = './result/img_ruxian/img(swin_unet)/'

    # label_path = './xunlianji//jiazhuangxian/test/label/'
    # predict_path = './result/img_jiazhuangxian/img(swin_unet)/'

    files = os.listdir(label_path)
    for filename in files:
        imgLabel = Image.open(label_path + filename)
        # imgPredict = Image.open(predict_path +filename.replace('.png','_res.png'))
        imgPredict = Image.open(predict_path + filename)

        img_GT = cv2.imread(label_path + filename, 0)
        img_PRE = cv2.imread(predict_path + filename, 0)
        ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret_R, binary_PRE = cv2.threshold(img_PRE, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imgPredict = np.array(imgPredict, dtype=np.uint8)  # 读取一张图像分割结果，转化成numpy数组
        imgLabel = np.array(imgLabel, dtype=np.uint8)  # 读取一张对应的标签，转化成numpy数组
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        # hist = metric.addBatch(imgPredict, imgLabel) # 打印混淆矩阵
        metric.addBatch(imgPredict, imgLabel)
        pa.append(metric.pixelAccuracy())
        # cpa.append(metric.classPixelAccuracy())
        # specificity.append(metric.Specificity())
        mpa.append(metric.meanPixelAccuracy())
        ioU.append(metric.IntersectionOverUnion())
        meanIoU.append(metric.meanIntersectionOverUnion())
        # dice.append(metric.Dice_coefficient())
        recall.append(metric.recall())
        precision.append(metric.precision())
        # F1_score.append(metric.F1_score())
        FWIOU.append(metric.Frequency_Weighted_Intersection_over_Union())
        DSI.append(np.float32(format(calDSI(binary_GT, binary_PRE))))
        VOE.append(np.float32(format(calVOE(binary_GT, binary_PRE))))
        RVD.append(np.float32(format(calRVD(binary_GT, binary_PRE))))
        # Precision.append(np.float32(format(calPrecision(binary_GT, binary_PRE))))
        # Recall.append(np.float32(format(calRecall(binary_GT, binary_PRE))))
    Pa = np.mean(pa)
    # Cpa = np.mean(cpa, axis=0)
    # specificity = np.mean(specificity, axis=0)
    Mpa = np.mean(mpa)
    IoU = np.mean(ioU,  axis=0)
    mIoU = np.mean(meanIoU)
    Dice = np.mean(dice, axis=0)
    recall = np.mean(recall,  axis=0)
    precision = np.mean(precision, axis=0)
    FWIOU = np.mean(FWIOU)
    DSI = np.mean(DSI)
    F1_score = 2 * recall * precision / (recall + precision)
    np.seterr(divide='ignore', invalid='ignore')
    VOE = np.mean(VOE)
    RVD = np.mean(RVD)
    # Precision = np.mean(Precision)
    # Recall = np.mean(Recall)
    # F1_score = (2 * Recall * Precision / (Recall + Precision))
    print('pa is : %f' % Pa)
    # print('cpa is :', Cpa)  # 列表
    # print('specificity is : ', specificity)
    print('mpa is : %f' % Mpa)
    print('IoU is :  ',   IoU)
    print('mIoU is  ', mIoU)
    # print('Dice is :', Dice)
    print('recall is : ',   recall)
    print('precision is : ', precision)
    print('F1_score is :', F1_score)
    print('FWIOU is :', FWIOU)
    print('DSI is :', DSI)
    print('VOE is :', VOE)
    print('RVD is : ',   RVD)
    # print('Precision is :', Precision)
    # print('Recall is :', Recall)


#===============================================================\
    # 计算单张图片评价指标
    # imgPredict = cv2.imread('262.png')
    # imgLabel = cv2.imread('262_mask.png')
    # imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
    # imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
    # metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
    # hist = metric.addBatch(imgPredict, imgLabel)
    # pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    # mpa = metric.meanPixelAccuracy()
    # IoU = metric.IntersectionOverUnion()
    # mIoU = metric.meanIntersectionOverUnion()
    # print('hist is :\n', hist)
    # print('PA is : %f' % pa)
    # print('cPA is :', cpa)  # 列表
    # print('mPA is : %f' % mpa)
    # print('IoU is : ', IoU)
    # print('mIoU is : ', mIoU)
#=======================================================