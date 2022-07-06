import numpy as np
import cv2

roi = cv2.imread('daun.jpg')
hsv = cv2.cvColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('')