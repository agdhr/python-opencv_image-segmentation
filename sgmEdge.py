import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
# https://learnopencv.com/edge-detection-using-opencv/

# CANNY EDGE DETECTION
# Read the original image
img = cv2.imread('lena.jpg',flags = 0)
# Blur the image for better edge detection
blur = cv2.GaussianBlur(img,(3,3),0)
# Canny edge detection
edges = cv2.Canny(image = blur, threshold1 = 100, threshold2 = 200)

# SOBEL EDGE DETECTION
# Laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)
# Sobel edge detection on the X axis
sobelx = cv2.Sobel(blur, ddepth = cv2.CV_64F, dx=1, dy=0, ksize = 5)
# Sobel edge detection on the Y axis
sobely = cv2.Sobel(blur, ddepth = cv2.CV_64F, dx=0, dy=1, ksize = 5)
# Sobel edge detection on the Y and Y axis
sobelxy = cv2.Sobel(blur, ddepth = cv2.CV_64F, dx=1, dy=1, ksize = 5)
# Display sobel edge detection images
result = np.hstack((laplacian,sobelx,sobely,sobelxy))
cv2.imshow('Sobel Edge Detection Images', result)
cv2.waitKey(0)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()