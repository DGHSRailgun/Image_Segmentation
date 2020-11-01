from skimage import data, color, filters
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cv2 import GaussianBlur


def sys_log(img):
    '''
    系统自带函数
    :param img: 待测图像
    :return: 返回边缘图像矩阵
    '''
    gray_img = cv2.imread(img, 0)  # 读取图片为灰色
    g_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0)  # 以核大小为3x3，方差为0的高斯函数进行高斯滤波
    system_edge_img = cv2.Laplacian(g_img, cv2.CV_16S, ksize=3)  # laplace检测
    system_edge_img = cv2.convertScaleAbs(system_edge_img)  # 转为8位
    return system_edge_img


def def_log(img):
    # 第一步灰度化
    gray_img = cv2.imread(img, 0)

    # 第二步高斯滤波
    # 高斯算子
    g_filter = np.array([[0, 0, 1, 0, 0],
                         [0, 1, 2, 1, 0],
                         [1, 2, 16, 2, 1],
                         [0, 1, 2, 1, 0],
                         [0, 0, 1, 0, 0]])
    self_g_img = np.pad(gray_img, ((2, 2), (2, 2)), 'constant')  # 扩展操作
    # 以下进行的其实就是滤波操作
    w, h = self_g_img.shape
    for i in range(w - 4):
        for j in range(h - 4):
            self_g_img[i][j] = np.sum(self_g_img[i:i + 5, j:j + 5] * g_filter)

    # 第三步：计算laplace二阶导数，操作和laplace算子一样
    lap4_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 4邻域laplacian算子
    g_pad = np.pad(self_g_img, ((1, 1), (1, 1)), 'constant')
    # 4邻域
    edge4_img = np.zeros((w, h))
    for i in range(w - 2):
        for j in range(h - 2):
            edge4_img[i, j] = np.sum(g_pad[i:i + 3, j:j + 3] * lap4_filter)
            if edge4_img[i, j] < 0:
                edge4_img[i, j] = 0  # 把所有负值修剪为0

    lap8_filter = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]])  # 8邻域laplacian算子
    # 8邻域
    g_pad = np.pad(self_g_img, ((1, 1), (1, 1)), 'constant')
    edge8_img = np.zeros((w, h))
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            edge8_img[i, j] = np.sum(g_pad[i - 1:i + 2, j - 1:j + 2] * lap8_filter)
            if edge8_img[i, j] < 0:
                edge8_img[i, j] = 0
    return [edge4_img, edge8_img]


if __name__ == '__main__':
    img = 'Pyramid.jpg'
    edge_img = sys_log(img)
    edge4_img, edge8_img = def_log(img)
    cv2.imshow('system_log', edge_img)
    cv2.imshow('self_log', edge4_img)
    cv2.waitKey(0)