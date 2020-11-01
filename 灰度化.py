import cv2
import numpy as np

img = cv2.imread('Pyramid.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height,width,3),np.uint8)
for i in range(height):
    for j in range(width):
        (b,g,r) = img[i,j]
        gray = (int(b) + int(g) +int(r))/3
        dst[i,j] = np.uint8(gray)
cv2.imshow('original',img)
cv2.imshow('image',dst)
cv2.waitKey(0)
