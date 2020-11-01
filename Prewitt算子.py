from skimage import data, color, filters
import matplotlib.pyplot as plt
import numpy as np
import cv2


def sys_prewitt(img):
    '''
    prewitt系统自带
    :param img: 原始图像
    :return: 返回边缘图像
    '''
    img = cv2.imread(img, 0)
    edge_img = filters.prewitt(img)
    return edge_img


def def_col(img, kernal):
    '''
    对指定的图像用指定的核对齐其进行卷积，得到卷积后的图像
    :param img: 补零操作后的灰度图像
    :param kernal: 卷积核
    :return: 返回卷积后的图像，即边缘图像
    '''
    edge_img = cv2.filter2D(img, -1, kernal)
    # h,w = img.shape
    # edge_img=np.zeros([h-2,w-2])
    # for i in range(h-2):
    #     for j in range(w-2):
    #         edge_img[i,j]=img[i,j]*kernal[0,0]+img[i,j+1]*kernal[0,1]+img[i,j+2]*kernal[0,2]+\
    #                 img[i+1,j]*kernal[1,0]+img[i+1,j+1]*kernal[1,1]+img[i+1,j+2]*kernal[1,2]+\
    #                 img[i+2,j]*kernal[2,0]+img[i+2,j+1]*kernal[2,1]+img[i+2,j+2]*kernal[2,2]
    return edge_img


def def_prewitt(img, type_flags):
    gray = cv2.imread(img, 0)
    h = gray.shape[0]
    w = gray.shape[1]
    x_prewitt = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
    y_prewitt = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])

    img = np.zeros([h + 2, w + 2])
    img[2:h + 2, 2:w + 2] = gray[0:h, 0:w]
    edge_x_img = def_col(img, x_prewitt)
    edge_y_img = def_col(img, y_prewitt)

    # p(i,j)=max[edge_x_img,edge_y_img]这里是将x,y中最大的梯度来代替该点的梯度
    edge_img_max = np.zeros([h, w], np.uint8)
    for i in range(h):
        for j in range(w):
            if edge_x_img[i][j] > edge_y_img[i][j]:
                edge_img_max = edge_x_img[i][j]
            else:
                edge_img_max = edge_y_img

    # p(i,j)=edge_x_img+edge_y_img#将梯度和替代该点梯度
    edge_img_sum = np.zeros([h, w], np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_sum[i][j] = edge_x_img[i][j] + edge_y_img[i][j]

    # p(i,j)=|edge_x_img|+|edge_y_img|将绝对值的和作为梯度
    edge_img_abs = np.zeros([h, w], np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_abs[i][j] = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])

    # p(i,j)=sqrt(edge_x_img**2+edge_y_img**2)将平方和根作为梯度
    edge_img_sqrt = np.zeros([h, w], np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_sqrt[i][j] = np.sqrt((edge_x_img[i][j]) ** 2 + (edge_y_img[i][j]) ** 2)

    type = [edge_img_max, edge_img_sum, edge_img_abs, edge_img_sqrt]
    return type[type_flags]


if __name__ == '__main__':
    img = 'Pyramid.jpg'
    edge = sys_prewitt(img)
    edge1 = def_prewitt(img, 3)
    cv2.imshow('system_definition', edge)
    cv2.imshow('self_definition', edge1)
    cv2.waitKey(0)