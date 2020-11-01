from skimage import data, color, filters
import numpy as np
import matplotlib.pyplot as plt
import cv2

gray_img = cv2.imread('Pyramid.jpg', 0)

# 先对图像进行两次高斯滤波运算
gimg1 = filters.gaussian(gray_img, sigma=2)
gimg2 = filters.gaussian(gray_img, sigma=1.6 * 2)

# 两个高斯运算的差分
dimg = gimg2 - gimg1

# 将差归一化
dimg /= 2

# cv2.imshow('', dimg)
# cv2.waitKey(0)
figure = plt.figure()
plt.subplot(141).set_title('original_img1')
plt.imshow(gray_img)
# cv2.imshow('gray_img',gray_img)
plt.subplot(142).set_title('LoG_img1')
plt.imshow(gimg1)
# cv2.imshow('gimg1',gimg1)
plt.subplot(143).set_title('LoG_img2')
plt.imshow(gimg2)
# cv2.imshow('gimg2',gimg2)
plt.subplot(144).set_title('DoG_edge')
plt.imshow(dimg, cmap='gray')
# cv2.imshow('dimg',dimg)
plt.show()
