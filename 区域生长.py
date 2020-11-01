import numpy as np
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

#计算像素之间的偏差
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

#设定八邻域
def selectConnects():
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                Point(0, 1), Point(-1, 1), Point(-1, 0)]  # 8邻域
    return connects

#生长函数
def regionGrow(img, seeds, thresh):
    # 读取图像的宽高，并建立一个和原图大小相同的seedMark
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    # 将定义的种子点放入种子点序列seedList
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects()
    # 逐个点开始生长，生长的结束条件为种子序列为空，即没有生长点
    while (len(seedList) > 0):
        # 弹出种子点序列的第一个点作为生长点
        currentPoint = seedList.pop(0)
        # 并将生长点对应seedMark点赋值label（1），即为白色
        seedMark[currentPoint.x, currentPoint.y] = label
        # 以种子点为中心，八邻域的像素进行比较
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            # 判断是否为图像外的点，若是则跳过。  如果种子点是图像的边界点，邻域点就会落在图像外
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            # 判断邻域点和种子点的差值
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            # 如果邻域点和种子点的差值小于阈值并且是没有被分类的点，则认为是和种子点同类，赋值label，
            # 并作为下一个种子点放入seedList
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


if __name__ == '__main__':
    img = cv2.imread('Pyramid.jpg', 0)
    ret1, th1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # 选定种子点
    seeds = [Point(10, 10)]
    # 开始从种子点生长
    binaryImg = regionGrow(th1, seeds, 10)
    cv2.imshow('Original', img)
    cv2.imshow('regionGrow', binaryImg)
    cv2.imwrite('regionGrow.jpg',binaryImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()