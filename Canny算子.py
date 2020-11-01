# coding=utf-8
import cv2
import numpy as np


def cv2_canny(img):
    '''
    由于Canny只能处理灰度图，所以将读取的图像转成灰度图。
    用高斯平滑处理原图像降噪。
    调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
    :param img:
    :return:
    '''
    img = cv2.imread(img, 0)
    img = cv2.GaussianBlur(img, (3, 3), 2)
    edge_img = cv2.Canny(img, 50, 100)
    return edge_img


def def_canny(img, top=1, buttom=1, left=1, right=1):
    '''
    自定义canny算子，这里可以修改的参数有填充的方式和值、高斯滤波是核的大小以及sigmaX的值
    :param img: 待测图片
    :param top: 图像上方填充的像素行数
    :param buttom: 图像下方填充的像素行数
    :param left: 图像左方填充的像素行数
    :param right: 图像右方填充的像素行数
    :return: [img1,img2,img3,theta]
    img1：梯度幅值图；img2：非极大值抑制梯度灰度图；img3：最终边缘图像；theta：梯度方向灰度图
    '''
    # 算子，这里的梯度算子可以使用sobel、sobel等算子，这里用的是prewitt梯度算子
    m1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    m2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # 第一步：完成高斯平滑滤波
    img = cv2.imread(img, 0)  # 将彩色读成灰色图片
    img = cv2.GaussianBlur(img, (3, 3), 2)  # 高斯滤波

    # 第二步：完成一阶有限差分计算，计算每一点的梯度幅值与方向
    img1 = np.zeros(img.shape, dtype="uint8")  # 与原图大小相同，用于存放梯度值
    theta = np.zeros(img.shape, dtype="float")  # 方向矩阵原图像大小，用于存放梯度方向
    img = cv2.copyMakeBorder(img, top, buttom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=0)  # 表示增加的像素数，类似于补零操作，这里上下左右均增加了一行填充方式为cv2.BORDER_REPLICATE
    rows, cols = img.shape

    # 以下是进行了卷积，以核与图像的卷积来代替核中间对应的那个值
    # dst = cv2.filter2D(img, -1, kernel)这个函数虽然可以直接进行卷积，但是无法返回梯度方向
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Gy，对应元素相乘再累加
            Gy = (np.dot(np.array([1, 1, 1]), (m1 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            # Gx，对应元素相乘再累加
            Gx = (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            # 将所有的角度转换到-180到180之间
            if Gx[0] == 0:
                theta[i - 1, j - 1] = 90
                continue
            else:
                temp = (np.arctan(Gy[0] / Gx[0])) * 180 / np.pi  # 求梯度方向
            if Gx[0] * Gy[0] > 0:
                if Gx[0] > 0:
                    theta[i - 1, j - 1] = np.abs(temp)
                else:
                    theta[i - 1, j - 1] = (np.abs(temp) - 180)
            if Gx[0] * Gy[0] < 0:
                if Gx[0] > 0:
                    theta[i - 1, j - 1] = (-1) * np.abs(temp)
                else:
                    theta[i - 1, j - 1] = 180 - np.abs(temp)
            img1[i - 1, j - 1] = (np.sqrt(Gx ** 2 + Gy ** 2))  # 求两个方向梯度的平方和根

    # 以下对梯度方向进行了划分，将梯度全部划分为0,90,45,-45四个角度
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):
            if (((theta[i, j] >= -22.5) and (theta[i, j] < 22.5)) or
                    ((theta[i, j] <= -157.5) and (theta[i, j] >= -180)) or
                    ((theta[i, j] >= 157.5) and (theta[i, j] < 180))):
                theta[i, j] = 0.0
            elif (((theta[i, j] >= 22.5) and (theta[i, j] < 67.5)) or
                  ((theta[i, j] <= -112.5) and (theta[i, j] >= -157.5))):
                theta[i, j] = 45.0
            elif (((theta[i, j] >= 67.5) and (theta[i, j] < 112.5)) or
                  ((theta[i, j] <= -67.5) and (theta[i, j] >= -112.5))):
                theta[i, j] = 90.0
            elif (((theta[i, j] >= 112.5) and (theta[i, j] < 157.5)) or
                  ((theta[i, j] <= -22.5) and (theta[i, j] >= -67.5))):
                theta[i, j] = -45.0

    # 第三步：进行 非极大值抑制计算
    # 这里做的事情其实就是在寻找极值点，像素点的值是与该像素点的方向垂直方向上相邻三个像素最大的，则将该像素点定义为极值点，即边缘像素
    img2 = np.zeros(img1.shape)  # 非极大值抑制图像矩阵
    for i in range(1, img2.shape[0] - 1):
        for j in range(1, img2.shape[1] - 1):
            if (theta[i, j] == 0.0) and (img1[i, j] == np.max([img1[i, j], img1[i + 1, j], img1[i - 1, j]])):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == -45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j - 1], img1[i + 1, j + 1]]):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == 90.0) and img1[i, j] == np.max([img1[i, j], img1[i, j + 1], img1[i, j - 1]]):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == 45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j + 1], img1[i + 1, j - 1]]):
                img2[i, j] = img1[i, j]

    # 第四步：双阈值检测和边缘连接
    img3 = np.zeros(img2.shape)  # 定义双阈值图像
    # TL = 0.4*np.max(img2)
    # TH = 0.5*np.max(img2)
    TL = 50  # 低阈值
    TH = 100  # 高阈值
    # 关键在这两个阈值的选择
    # 小于低阈值的则设置为0，大于高阈值的则设置为255
    for i in range(1, img3.shape[0] - 1):
        for j in range(1, img3.shape[1] - 1):
            # 将比较明确的灰度级的像素点绘制出来
            if img2[i, j] < TL:
                img3[i, j] = 0
            elif img2[i, j] > TH:
                img3[i, j] = 255
            # 如果一个像素点的值在二者之间，且这个像素点的邻近的八个像素点有一个的像素值是小于高阈值的，则将该点设置为255
            # 实际上下面这段代码就是将剩余的像素点都设置成了255，这样就实现了边缘连接
            elif ((img2[i + 1, j] < TH) or (img2[i - 1, j] < TH) or (img2[i, j + 1] < TH) or
                  (img2[i, j - 1] < TH) or (img2[i - 1, j - 1] < TH) or (img2[i - 1, j + 1] < TH) or
                  (img2[i + 1, j + 1] < TH) or (img2[i + 1, j - 1] < TH)):
                img3[i, j] = 255
    return [img1, img2, img3, theta]


if __name__ == '__main__':
    img = 'Pyramid.jpg'
    img1, img2, img3, theta = def_canny(img)
    cv2.imshow("Origin_gray", cv2.imread(img, 0))  # 原始图像
    cv2.imshow("gradient_magnitude", img1)  # 梯度幅值图
    cv2.imshow("Non_maximum_suppression_gray", img2)  # 非极大值抑制灰度图
    cv2.imshow("Final_result", img3)  # 最终效果图
    cv2.imshow("Degree_gray", theta)  # 角度值灰度图
    edge_img = cv2_canny(img)
    cv2.imshow("origin", cv2.imread(img))
    cv2.imshow('CV2_Canny', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()