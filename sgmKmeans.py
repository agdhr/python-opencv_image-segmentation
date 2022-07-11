# K-Means Clustering for Image Segmentation using OpenCV in Python
# https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3

# Loading required libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('town.png')   # Loading image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change color to RGB (from BGR)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixvals = image.reshape((-1,3))
# Convert to float type only for supporting cv2.Kmean
pixvals = np.float32(pixvals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 4
retval, labels, centers = cv2.kmeans(pixvals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
sedmented_data = centers[labels.flatten()]
segmended_image = sedmented_data.reshape((image.shape))
plt.imshow(segmended_image)
cv2.imshow('Segmented image', segmended_image)
cv2.waitKey(0)