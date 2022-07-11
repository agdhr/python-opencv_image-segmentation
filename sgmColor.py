# SEGMENTASI SPO
# Tujuan untuk mempartisi gambar menjadi beberapa wilayah yang tidak tumpang tindih dengan karakteristik
# yang homogen, seperti intensitas, warna, dan tekstur.

# K-Means Algorithm
# is a clustering algorithm that used to group data points into clusters such that data points lying in the same group
# are very similar to each other in characteristics
import matplotlib as plt
import numpy as n
import cv2

# Contour Detection
# is defined as curves/polygons formed by joining the pixels that are grouped together according to intensity or color values

# Masking
# masks (are binary images, only 0 and 1 as pixel values) to transform picture is no


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

# K-MEANS ALGORITHM
if not image is None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to np.float32
    twoDimage = rgb.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    #define criteria, number of cluster (K), and apply Kmeans ()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6       # coba ganti-ganti nilai K untuk hasil yang berbeda
    attempts = 10
    # Now convert back into uint8, and make origical image
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria,attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    cv2.imshow('res', result_image)
    cv2.waitKey(0)

# CONTOUR DETECTION
if not image is None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    th, biner = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    plt.imshow(biner, cmap="gray")
    plt.show()
    #edges = cv2.dilate(cv2.Canny(biner, 0, 255), None)
    #cv2.imshow('res2', edges)
    #cv2.waitKey(0)

citra = cv2.imread('infectedleaf2.jpg',0)
if not citra is None:
    citra = 1 - citra
    th, citraBiner = cv2.threshold(citra, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    jumlahBaris = citraBiner.shape[0]  # memperoleh jumlah baris berwarna hitam
    jumlahKolom = citraBiner.shape[1]  # memperoleh jumlah baris berwarna putih
    citraKontur = np.zeros((jumlahBaris, jumlahKolom, 3), np.uint8)

    kontur, hierarki = cv2.findContours(citraBiner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    citraRGB = cv2.merge((citraBiner, citraBiner, citraBiner))
    hasil = np.hstack((citraRGB, citraKontur))
    cv2.imshow('Hasil', hasil)
    cv2.waitKey()
# COLOR MASKING
if not image is None:
    # Konversi ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mencari batas rendah dan batas atas RGB dari WARNA BIRU
    greenLight = np.array([30, 40, 20])
    greenMedium = np.array([179, 255, 255])

    # Peroleh bagian yang berwarna biru
    mask = cv2.inRange(hsv, greenLight, greenMedium)

    # Kenakan operasi "dan" terhadap citra asli
    healthyArea = cv2.bitwise_and(image, image, mask = mask)

    spot = image - healthyArea

    #cv2.imshow('Original Image', citra)
    cv2.imshow('Masked Image', mask)
    cv2.imshow('Healthy Leaf', healthyArea)
    cv2.imshow('Infected area', spot)

    cv2.waitKey(0)

