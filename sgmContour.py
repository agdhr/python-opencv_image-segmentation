# IMAGE SEGMENTATION
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/#4_Segmentation_using_Color_Masking

# Importing libraries and images
import cv2
import matplotlib.pyplot as plt
import numpy as np
path = 'orange.jpg'
image = cv2.imread(path)
image = cv2.resize(image, (256,256))

# Preprocessing the image
# --- convert the image to grayschale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# --- compute the threshold of the grayscale image
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
# --- apply canny edge detection to the thresholded image before finally using the ‘cv2.dilate’ function to dilate edges detected.
edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

# Detecting and Drawing Contours
# --- Use the OpenCV find contour function to find all the open/closed regions in the image and store (cnt). Use the -1 subscript since the function returns a two-element tuple
# --- Pass them through the sorted function to access the largest contours first
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
# --- Create a zero pixel mask that has equal shape and size to the original image
mask = np.zeros((256,256), np.uint8)
# --- Draw the detected contours on the created mask.
masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

# Segmenting the regions
# --- perform a bitwise AND operation on the original image (img) and the mask (containing the outlines of all our detected contours).
dst = cv2.bitwise_and(image, image, mask = mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

cv2.imshow('Segmented image', segmented)
cv2.waitKey(0)
