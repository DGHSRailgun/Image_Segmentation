import cv2
from skimage import filters, color
import numpy as np


def cv2_sobel(img):
    '''
    利用cv2自带的函数进行sobel边缘检测
    :param img: 待检测的图像
    :return: 返回不同方向梯度的边缘图像矩阵，通过控制不同方向的导数为1或者0来选择卷积核
    '''
    img = cv2.imread(img)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0)  # 通过控制dx,dy的值来控制梯度的方向，这里是求x方向的梯度
    sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1)  # 这里是求y方向的梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1)  # 这里是求x,y方向综合的梯度
    return [sobel_x, sobel_y, sobel]


def skimage_sobel(img):
    '''
    利用skimage模块自带的函数进行图像的边缘检测
    :param img: 待检测图像
    :return: 返回的是边缘图像矩阵
    '''
    gray = cv2.imread(img, 0)
    edge_img = filters.sobel(gray)  # 这里求的是x,y方向两个梯度的综合
    return edge_img


def sobel_cal(img, kernal):
    '''
    自定义sobel卷积函数
    :param img: 已经补零了的图像
    :param kernal: 指定方向的梯度算子，核
    :return: 返回梯度矩阵
    '''
    edge_img = cv2.filter2D(img, -1, kernal)
    # h, w = img.shape
    # img_filter = np.zeros([h, w])
    # #机芯卷积操作
    # for i in range(h - 2):
    #     for j in range(w - 2):
    #         img_filter[i][j] = img[i][j] *kernal[0][0] + img[i][j + 1] * kernal[0][1] + img[i][j + 2] * kernal[0][2] + \
    #                                img[i + 1][j] * kernal[1][0] + img[i + 1][j + 1] *kernal[1][1] + img[i + 1][j + 2] * kernal[1][2] + \
    #                                img[i + 2][j] * kernal[2][0] + img[i + 2][j + 1] * kernal[2][1] + img[i + 2][j + 2] * kernal[2][2]
    return edge_img


def def_sobel(img):
    # 定义不同方向的梯度算子
    x_sobel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    y_sobel = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    gray_img = cv2.imread(img, 0)
    h, w = gray_img.shape
    img = np.zeros([h + 2, w + 2], np.uint8)
    img[2:h + 2, 2:w + 2] = gray_img[0:h, 0:w]
    x_edge_img = sobel_cal(img, x_sobel)
    y_edge_img = sobel_cal(img, y_sobel)
    edge_img = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            edge_img[i][j] = np.sqrt(x_edge_img[i][j] ** 2 + y_edge_img[i][j] ** 2) / (np.sqrt(2))
    return edge_img


if __name__ == '__main__':
    img = "Pyramid.jpg"
    sobel_x, sobel_y, sobel = cv2_sobel(img)
    sk_sobel = skimage_sobel(img)
    def_sobelimg = def_sobel(img)
    cv2.imshow("src", cv2.imread(img))
    cv2.imshow("CV_Sobel_x", sobel_x)
    cv2.imshow("CV_Sobel_y", sobel_y)
    cv2.imshow("CV_Sobel", sobel)
    cv2.imshow("SK_system_definition", sk_sobel)
    cv2.imshow("self_definition", def_sobelimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()