# THRESHOLD-BASED SEGMENTATION
# Thresholding is the simplest method of image segmentation. It is a non-linear operation that
# converts a gray-scale image into a binary image where the two levels are assigned to pixels
# that are below or above the specified threshold value. In other words, if pixel value is greater
# than a threshold value, it is assigned one value (maybe white), else it is assigned another
# value (maybe black).

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read an image and convert to Grayscale
img = cv2.imread('circle.png',0)

# THRESHOLDING
# cv2.threshold(src, thresh, maxval, type[, dst])
# src = input image
# thresh = threshold value, and it is used to classify the pixel values.
# maxval = represents the value to be given if pixel value is more than (sometimes less than) the threshold value.
# type =  thresholding type

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original game', 'Binary', 'Binary Inv', 'Truncy', 'Tozeroy', 'Tozeroy Inv']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# ADAPTIVE THRESHOLDING
# cv.AdaptiveThreshold(src, dst, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1=5)
# adaptiveMethod = Adaptive thresholding algorithm to use
# thresholdType must be THRESH_BINARY or THRESH_BINARY_INV
# blockSize = size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# C - Constant subtracted from the mean or weighted mean

citra = cv2.imread('bw.png', 0)
citra = cv2.medianBlur(citra, 5)

ret, thresh6 = cv2.threshold(citra, 127, 255, cv2.THRESH_BINARY)
thresh7 = cv2.adaptiveThreshold(citra, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh8 = cv2.adaptiveThreshold(citra, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, thresh6, thresh7, thresh8]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
