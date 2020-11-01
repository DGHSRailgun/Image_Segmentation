import numpy as np
import matplotlib.pyplot as plt
import cv2


def sys_laplace(img):
    '''
    调用系统函数进行图像的边缘二阶算子laplace检测
    :param img: 待测图像
    :return: 返回的是边缘图像矩阵
    '''
    gray = cv2.imread(img, 0)  # 读取图片为灰色
    edeg_img = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)  # 将原始图片进行边缘检测，边缘图像放在16位中防止溢出，使用的卷积核是3
    edge_img = cv2.convertScaleAbs(
        edeg_img)  # 转回8位图像，brief Scales, calculates absolute values, and converts the result to 8-bit
    return edge_img


def def_laplace4(img):
    '''
    自定义二阶算子边缘检测4邻域laplace
    :param img:待测图像
    :return: 返回4邻域边缘检测的图像矩阵
    '''
    gray_img = cv2.imread(img, 0)
    w, h = gray_img.shape
    # 在灰度图像的四周填充一行/一列0,卷积核为3x3,要想对原始图像的每一个像素进行卷积则需要进行填充
    ori_pad = np.pad(gray_img, ((1, 1), (1, 1)), 'constant')
    # 定义两个不同的laplace算子
    # 不同的算子涉及到的邻域像素点的个数也不一样
    lap4_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 4邻域laplacian算子

    # 4邻域
    edge4_img = np.zeros((w, h))
    for i in range(w - 2):
        for j in range(h - 2):
            edge4_img[i, j] = np.sum(ori_pad[i:i + 3, j:j + 3] * lap4_filter)  # 进行卷积
            if edge4_img[i, j] < 0:
                edge4_img[i, j] = 0  # 把所有负值修剪为0
    edge4_img = cv2.convertScaleAbs(edge4_img)  # 将图像变成8位的图像，其实就是一个映射关系，类似于均衡化中的映射关系
    return edge4_img


def def_laplace8(img):
    '''
    自定义二阶算子边缘检测8邻域laplace
    :param img:待测图像
    :return: 返回8邻域边缘检测的图像矩阵
    '''
    gray_img = cv2.imread(img, 0)
    w, h = gray_img.shape
    # 在灰度图像的四周填充一行/一列0,卷积核为3x3,要想对原始图像的每一个像素进行卷积则需要进行填充
    ori_pad = np.pad(gray_img, ((1, 1), (1, 1)), 'constant')
    # 定义两个不同的laplace算子
    # 不同的算子涉及到的邻域像素点的个数也不一样
    lap8_filter = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]])  # 8邻域laplacian算子

    # 8邻域
    edge8_img = np.zeros((w, h))
    for i in range(w - 2):
        for j in range(h - 2):
            edge8_img[i, j] = np.sum(ori_pad[i:i + 3, j:j + 3] * lap8_filter)  # 进行卷积
            if edge8_img[i, j] < 0:
                edge8_img[i, j] = 0
    edge8_img = cv2.convertScaleAbs(edge8_img)
    return edge8_img


if __name__ == '__main__':
    img = 'Pyramid.jpg'
    sys_img = sys_laplace(img)
    edge4_img = def_laplace4(img)
    edge8_img = def_laplace8(img)
    cv2.imshow('sys', sys_img)
    cv2.imshow('def3', edge4_img)
    cv2.imshow('def4', edge8_img)
    cv2.waitKey(0)