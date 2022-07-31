# Segmentation using Color Masking
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/#4_Segmentation_using_Color_Masking

import numpy as n
import cv2
# Color detection
# classification of colors by using ther RGB colorspace values.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# PRE-PROCESSING

# Step 1 --- Load image
image = cv2.imread('infectedleaf2.jpg')
h,w,c = image.shape
print(h,"px", w, "px")
# Stel 2 --- Resizing image
image = cv2.resize(image,(1000,600), interpolation = cv2.INTER_NEAREST)
# Step 3 --- Noise Reducing
imgBlur = cv2.medianBlur(image, 3)
# Step 4 --- Background Removal
# COLOR MASKING
# Konversi ke HSV
hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# SEGMENTASI OBJEK DENGAN BACKGROUND
# ----------------------------------------------------------------------------------------------------------------------
# Mencari batas rendah dan batas atas RGB dari WARNA HSV
lower1 = np.array([0, 35, 0])
upper1 = np.array([179, 255, 255])
# Peroleh bagian background yang berwarna putih
mask1 = cv2.inRange(hsvimg, lower1, upper1)
# Kenakan operasi "dan" terhadap citra asli
leafregion = cv2.bitwise_and(image, image, mask = mask1)
background = image - leafregion
#cv2.imshow('Original Image', citra)
cv2.imshow('Masked Image', mask1)
cv2.imshow('Healthy Leaf', leafregion)
cv2.imshow('Infected area', background)
cv2.waitKey(0)
# cv2.imwrite('leafregion.png', leafregion)

# SEGMENTASI AREA DAUN SEHAT DENGAN SPOT (DAUN TERINFEKSI)
# Konversi ke HSV
hsvleaf = cv2.cvtColor(leafregion, cv2.COLOR_BGR2HSV)
lower2 = np.array([32,0,0])
upper2 = np.array([179,255,255])
mask2 = cv2.inRange(hsvleaf, lower2, upper2)
healthyArea = cv2.bitwise_and(leafregion, leafregion, mask = mask2)
spot = leafregion - healthyArea
cv2.imshow('Masked Image', mask2)
cv2.imshow('Healthy Leaf', healthyArea)
cv2.imshow('Infected area', spot)
cv2.imwrite('spot.png', spot)
cv2.waitKey(0)

