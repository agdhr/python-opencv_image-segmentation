# Segmentation using Color Masking
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/#4_Segmentation_using_Color_Masking

import cv2
import numpy as np

path = 'orange.jpg'
img = cv2.imread(path)

hvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([13, 13, 80])
upper = np.array([179, 255, 255])

mask = cv2.inRange(hvs, lower,upper)
result = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
