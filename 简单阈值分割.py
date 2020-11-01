# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('Pyramid.jpg', 0)
# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # binary （黑白二值）
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # （黑白二值反转）
# ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)  # 得到的图像为多像素值
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # 高于阈值时像素设置为255，低于阈值时不作处理
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)  # 低于阈值时设置为255，高于阈值时不作处理
#
# print(ret)
#
# cv2.imshow('Original Grey_Image', img)
# cv2.imshow('BINARY', thresh1)
# cv2.imshow('BINARY_INV', thresh2)
# cv2.imshow('TRUNC', thresh3)
# cv2.imshow('TOZERO', thresh4)
# cv2.imshow('TOZERO_INV', thresh5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Pyramid.jpg', 0)

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
cv.imwrite('BINARY.jpg',thresh1)
cv.imwrite('BINARY_INV.jpg',thresh2)
cv.imwrite('TRUNC.jpg',thresh3)
cv.imwrite('TOZERO.jpg',thresh4)
cv.imwrite('TOZERO_INV.jpg',thresh5)
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    title = titles[i]
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()