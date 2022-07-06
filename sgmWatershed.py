# WATERSHED SEGMENTATION
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation.php

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('coins.jpg')
b,g,r = cv2.split(img)
rgb = cv2.merge([r,g,b])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + +cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((2,2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)

plt.subplot(131), plt.imshow(rgb)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(thresh, 'gray')
plt.title("Otsu binary threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(thresh, 'gray')
plt.title("MorhologyEx"), plt.xticks([]), plt.yticks([])
plt.show()
