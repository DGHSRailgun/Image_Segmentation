# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# img = cv2.imread('Pyramid.jpg', 0)
# ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 简单滤波
# ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
# # 高斯滤波后再采用Otsu阈值
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(ret2)
# print(ret3)
# cv2.imshow('Original Grey_Image', img)
# cv2.imshow('BINARY', th1)
# cv2.imshow('OTSU', th2)
# cv2.imshow('OTSU', th2)
# # 用于解决matplotlib中显示图像的中文乱码问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.hist(img.ravel(), 256)
# plt.title('灰度直方图')
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Pyramid.jpg', 0)

print(img)
# 全局阈值
ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# Otsu阈值
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# ret2, th2 = cv.threshold(img, 0, 255, 1580)
print(ret2)
# 高斯滤波后再采用Otsu阈值
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# write img
cv.imwrite('Global-threshold.jpg',th1)
cv.imwrite('Otsu-threshold.jpg',th2)
cv.imwrite('Gaussian-Otsu-threshold.jpg',th3)

# 绘制所有图像及其直方图
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()