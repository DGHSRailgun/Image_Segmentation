import cv2
import numpy as np

threshold = 127  # 此处更改像素阈值，0为黑色，255为白色
img = cv2.imread('111.jpg')  # 此处更改文件名
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh1, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow('result', result)
cv2.waitKey(0)

black = np.sum(result < threshold)
length, width = result.shape

print('the size of picture is {length} * {width}'.format(length=length, width=width))
print('the sum of pixel is {sum}'.format(sum=length * width))
print('the sum of black pixel is {black}'.format(black=black))
print('proportion is {}'.format(black / (length * width)))
