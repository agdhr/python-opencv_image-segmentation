# Segmentation using Color Masking
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/#4_Segmentation_using_Color_Masking

import cv2
path = 'beach.png'
img = cv2.imread(path)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hvs = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

light_blue = (90, 70, 50)
dark_blue = (128, 255, 255)

# You can use the following values for green
# light_green = (40, 40, 40)
# dark_greek = (70, 255, 255)

mask = cv2.inRange(hvs, light_blue,dark_blue)
result = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('Hasil', result)
cv2.waitKey(0)