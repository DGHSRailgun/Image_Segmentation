import cv2
import numpy as np
from skimage import data, color, filters

def def_col(gray ,kernal):
    '''
    对指定的图像用指定的核对齐其进行卷积，得到卷积后的图像
    :param gray: 补零操作后的灰度图像
    :param kernal: 卷积核
    :return: 返回卷积后的图像，即边缘图像
    '''
    edge_img = cv2.filter2D(gray, -1, kernal)
    # 上面这句和下面的代码有一样的效果
    # h,w = gray.shape
    # edge_img = np.zeros((h-1,w-1))#用于存放卷积后的边缘图像矩阵
    # for i in range(h-1):
    #     for j in range(w-1):
    #         #用卷积核与图像上对应大小的块进行卷积，这里是以左上角的格子为基准进行的卷积
    #         edge_img[i,j]=gr  ay[i,j]*kernal[0,0]+gray[i,j+1]*kernal[0,1]+gray[i+1,j]*kernal[1,0]+gray[i,j]*kernal[1,1]
    return edge_img

def def_robert(img):
    '''
    对指定的图像进行边缘检测
    :param img: 要检测的图像
    :return: 返回边缘图像
    '''
    gray = cv2.imread(img ,0)
    # np.savetxt("result2.txt", gray, fmt="%d")
    h = gray.shape[0]
    w = gray.shape[1]
    # 定义Robert算子的卷积核,这个是对角线方向
    x_kernal = np.array([[1 ,0],
                         [0 ,-1]])
    y_kernal = np.array([[0 ,1],
                         [-1 ,0]])
    # 由于卷积核和图像进行卷积是以右下角的像素进行定位卷积核和目标像素点的位置关系，因此为了能够遍历整个图像，
    # 需要在图像的第一行和第一列进行补零操作
    gray_zero = np.zeros(( h +1 , w +1)  )  # 先生成一个(h+1,w+1)的零图像矩阵
    # 将原始图像去填充零矩阵，第一行和第一列不填充
    for i in range(1 , h +1):
        for j in range(1 , w +1):
            gray_zero[i ,j ] =gray[ i -1 , j -1]
    gray = gray_zero  # 将填充后的矩阵复制给gray
    # 通过卷积，得到x和y两个方向的边缘检测图像
    x_edge = def_col(gray ,x_kernal)
    y_edge = def_col(gray ,y_kernal)
    # 创建一个与原始图像大小一致的空图像，用于存放经过Robert算子的边缘图像矩阵
    edge_img = np.zeros((h ,w) ,np.uint8)
    # 根据计算公式得到最终的像素点灰度值
    for i in range(h):
        for j in range(w):
            edge_img[i ,j] = (np.sqrt(x_edge[i ,j ]** 2 +y_edge[i ,j ]**2))
    return edge_img

def sys_robert(img):
    '''
    直接调用系统skimage模块给的函数
    :param img: 待处理图片
    :return: 返回的是边缘图像矩阵
    '''
    gray = cv2.imread(img,0)#读取的时候直接将其读取成灰色的图片，变成一个二维灰度值矩阵
    edge_img = filters.roberts(gray)#进行边缘检测，得到边缘图像
    return edge_img


if __name__ == '__main__':
    img = "Pyramid.jpg"
    edge = def_robert(img)
    edge1 = sys_robert(img)
    # 以下是将看一下两种方法图像各点的像素值的比值，用c存放
    # h,w = edge1.shape
    # c = np.zeros((h,w))
    # c[0:h,0:w] = edge1[0:h,0:w]/edge1[0:h,0:w]
    # for i in range(h):
    #     for j in range(w):
    #         if edge1[i,j] != 0:
    #             c[i,j] = edge[i,j]/edge1[i,j]
    #         else:
    #             c[i,j] = edge[i,j]
    # print(max(c.ravel()))
    # 进行numpy保存到TXT中的操作
    # np.savetxt("result.txt", edge, fmt="%d")
    # np.savetxt("c.txt", c)
    # np.savetxt("result1.txt", edge1)
    cv2.imshow('self_definition' ,edge)
    cv2.imshow('system_definition' ,edge1)
    cv2.waitKey(0)