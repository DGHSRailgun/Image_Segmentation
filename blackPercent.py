# Auther: Yuanshi Li, Dongliang Ji
# Illustration: This python source code CANNOT be used in bussiness. Please visit 

import cv2
import numpy as np
import os


if __name__ == '__main__':
    filepath = input("请输入文件路径（按<Enter>确定）: ")


    threshold = 127  # 此处更改像素阈值，0为黑色，255为白色
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    # img = cv2.imread(temp)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh1, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('result', result)
    cv2.imwrite("result.jpg", result)
    print("关闭图片，自动保存图片到当前目录，并且计算处理结果...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    black = np.sum(result < threshold)
    length, width = result.shape

    print('the size of picture is {length} * {width}'.format(length=length, width=width))
    print('the sum of pixel is {sum}'.format(sum=length * width))
    print('the sum of black pixel is {black}'.format(black=black))
    print('proportion is {}'.format(black / (length * width)))
    os.system('pause')
