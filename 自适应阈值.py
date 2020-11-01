# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('Pyramid.jpg', 0)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 第一个参数为原始图像矩阵，第二个参数为像素值上限，第三个是自适应方法（adaptive method）：
# #                                              -----cv2.ADAPTIVE_THRESH_MEAN_C:领域内均值
# #                                              -----cv2.ADAPTIVE_THRESH_GAUSSIAN_C:领域内像素点加权和，权重为一个高斯窗口
# # 第四个值的赋值方法：只有cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV
# # 第五个Block size：设定领域大小（一个正方形的领域）
# # 第六个参数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
# # 这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图都用一个阈值
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('Original Grey_Image', img)
# cv2.imshow('BINARY', th1)
# cv2.imshow('MEAN_C_5', th2)
# cv2.imshow('MEAN_C_11', th3)
# cv2.imshow('GAUSSIAN_C_11', th4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt

literacy_path = 'Pyramid.jpg'

img_literacy = cv2.imread(literacy_path, 0)

# threshold
ret, thresh = cv2.threshold(img_literacy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)  # 124
# adaptive threshold
thresh1 = cv2.adaptiveThreshold(img_literacy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
thresh2 = cv2.adaptiveThreshold(img_literacy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh3 = cv2.adaptiveThreshold(img_literacy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
thresh4 = cv2.adaptiveThreshold(img_literacy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# write img
cv2.imwrite('otsu.jpg',thresh)
cv2.imwrite('adaptive_mean_0.jpg',thresh1)
cv2.imwrite('adaptive_mean_2.jpg',thresh2)
cv2.imwrite('adaptive_gaussian_0.jpg',thresh3)
cv2.imwrite('adaptive_gaussian_2.jpg',thresh4)

# show image
plt.figure('adaptive threshold', figsize=(12, 8))
plt.subplot(231), plt.imshow(img_literacy, cmap='gray'), plt.title('original')
plt.subplot(234), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
plt.subplot(232), plt.imshow(thresh1, cmap='gray'), plt.title('adaptive_mean_0')
plt.subplot(235), plt.imshow(thresh2, cmap='gray'), plt.title('adaptive_mean_2')
plt.subplot(233), plt.imshow(thresh3, cmap='gray'), plt.title('adaptive_gaussian_0')
plt.subplot(236), plt.imshow(thresh4, cmap='gray'), plt.title('adaptive_gaussian_2')

plt.show()

