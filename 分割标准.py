import cv2
from matplotlib import pyplot as plt
import numpy as np
from hausdorff import hausdorff_distance


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
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1
    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1
    Recall = R_s / R_t
    return Recall


def calTPR(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    t_s, t_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                t_s += 1
            if binary_GT[i][j] == 255:
                t_t += 1
    TPR = t_s / t_t
    return TPR


def calFPR(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    f_s, f_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 0 and binary_R[i][j] == 255:
                f_s += 1
            if binary_GT[i][j] == 0:
                f_t += 1
    FPR = f_s / f_t
    return FPR




if __name__ == '__main__':
    # step 1：读入图像，并灰度化
    img_GT = cv2.imread('Pyramid-GT.jpg', 0)
    img_R = cv2.imread('regionGrow.jpg', 0)
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # 灰度化
    # img_GT = img_GT[:,:,[2, 1, 0]]
    # img_R  = img_R[:,: [2, 1, 0]]

    # step2：二值化
    # 利用大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #判断二值图是否颜色相反
    if calPrecision(binary_GT, binary_R) < 0.25:
        binary_R = 255 - binary_R
    # cv2.imshow('binary_GT',binary_GT)
    # cv2.imshow('binary_R',binary_R)

    # step 3： 显示二值化后的分割图像与真值图像
    # plt.figure()
    # plt.subplot(121), plt.imshow(binary_GT), plt.title('真值图')
    # plt.axis('off')
    # plt.subplot(122), plt.imshow(binary_R), plt.title('分割图')
    # plt.axis('off')
    # plt.show()




    # step 4：计算DSI
    print('（1）Result of DICE，      DSI       = {0:.4}'.format(calDSI(binary_GT, binary_R)))  # 保留四位有效数字

    # step 5：计算VOE
    print('（2）Result of VOE，       VOE       = {0:.4}'.format(calVOE(binary_GT, binary_R)))

    # step 6：计算RVD
    print('（3）Result of RVD，       RVD       = {0:.4}'.format(calRVD(binary_GT, binary_R)))

    # step 7：计算TPR
    print('（4）Result of TPR，      TPR       = {0:.4}'.format(calTPR(binary_GT, binary_R)))

    # step 8：计算FPR
    print('（5）Result of FPR，      FPR       = {0:.4}'.format(calFPR(binary_GT, binary_R)))

    # step 9：计算Precision
    print('（6）Result of Precision， Precision = {0:.4}'.format(calPrecision(binary_GT, binary_R)))

    # step 10：计算Recall
    print('（7）Result of Recall，    Recall    = {0:.4}'.format(calRecall(binary_GT, binary_R)))