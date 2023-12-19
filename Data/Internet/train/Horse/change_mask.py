#处理GroundTruth中mask的问题
import cv2 as cv
import os
def change(name):
    I = cv.imread('./UsedImage/'+name, cv.IMREAD_GRAYSCALE)
    _,I=cv.threshold(I, 0, 255, cv.THRESH_BINARY)

    cv.imwrite('./GroundTruth_new/'+name, I)

imglist = os.listdir('./GroundTruth')

for i in range(len(imglist)):
    change(imglist[i])