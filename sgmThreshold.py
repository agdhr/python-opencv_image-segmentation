# IMAGE SEGMENTATION USING THRESHOLDING METHODS
# https://medium.com/swlh/image-processing-with-python-image-segmentation-using-thresholding-methods-423ecdaf8ab4

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import area_opening
from skimage.exposure import histogram
from skimage.filters import threshold_otsu
# Load and show the image
path = 'orangeplant.jpg'
img = imread(path)
imshow(img)
plt.show()
# Use trial and error to find thr arbitrary threshold value that best captures our desired shape of the objects.
th_values = np.linspace(0, 1, 11)   # threshold value
fig, axis = plt.subplots(2, 5, figsize = (15,8))
img_gray = rgb2gray(img)    # convert RGB to GrayScale
for th, ax in zip (th_values, axis.flatten()):
    img_binary = img_gray < th
    ax.imshow(img_binary)
    ax.set_title('$Threshold = %.2f$' % th)
plt.show()
# we can automate determining the thresholding value by looking at the intensity values
freq, bins = histogram(img_gray)
plt.step(bins, freq*1.0/freq.sum())
plt.xlabel('Intensity value')
plt.ylabel('Fraction of pixels')
plt.show()

# Otsu’s method assumes that the image is composed of a background and a foreground.
# This method works by minimizing the intra-class variance or maximizing the inter-class variance.
def masked_image (image, mask):
    r = img[:, :, 0] * mask
    g = img[:, :, 1] * mask
    b = img[:, :, 2] * mask
    return np.dstack([r,g,b])
fig, ax = plt.subplots(1, 2, figsize = (12,6))

thresh = threshold_otsu(img_gray)
img_otsu = img_gray < thresh
ax[0].imshow(img_otsu)
filtered = masked_image(img, img_otsu)
ax[1].imshow(filtered)
plt.show()
# This method for the assumption: the image is composed of a background and a foreground.

# most of our image segmentation problems is not a background-foreground problem.
# For example, we cannot simply use Otsu’s method in segmenting the Chico fruits from the leaves —
# this is because both are in the foreground of the image.
fig, ax = plt.subplots(1, 3, figsize = (15,6))
ax[0].imshow(img[:,:,0], cmap='Reds')
ax[0].set_title('Red')
ax[1].imshow(img[:,:,1], cmap='Greens')
ax[1].set_title('Green')
ax[2].imshow(img[:,:,2], cmap='Blues')
ax[2].set_title('Blue')
plt.show()

# Notice how the background has a high-intensity value while the objects themselves have a low-intensity value in all of
# the channels. This is because the background is the sky — which technically is the source of light. Thus, it naturally
# has a higher intensity value. This makes segmenting the image to background-foreground easier. However, this also makes
# segmenting images in the foreground much more challenging because of the much tighter pixel intensity range.
figure, ax = plt.subplots(1,2, figsize=(12,6))
chico_red = img[:,:,0]
chico_green = img[:,:,1]
chico_blue = img[:,:,2]
binarized = ((chico_red < 200) & (chico_red > 75) & (chico_green < 120) & (chico_green > 50) &
             (chico_blue > 20))
opened = area_opening(binarized, 5000)
ax[0].imshow(binarized)
ax[1].imshow(masked_image(img, opened))
plt.show()

# Difficult, right? Even if we used area_opening to clean the image, it still does not show good results.
# Now, let’s try if we can make this easier by using the HSV color space.
chico_hsv = rgb2hsv(img)
fig, ax = plt.subplots(1, 3, figsize=(15,6))
ax[0].imshow(chico_hsv[:,:,0], cmap='hsv')
ax[0].set_title('Hue')
ax[1].imshow(chico_hsv[:,:,1], cmap='hsv')
ax[1].set_title('Saturation')
ax[2].imshow(chico_hsv[:,:,2], cmap='hsv')
ax[2].set_title('Value')
plt.show()
# Notice how the HSV color space can visualize the image’s objects much better than the RGB color space.
# Let’s try to use thresholding methods using this color space.
figure, ax = plt.subplots(1,2, figsize=(12,6))
chico_hue = chico_hsv[:,:,0]
chico_sat = chico_hsv[:,:,1]
chico_val = chico_hsv[:,:,2]
binarized_hsv = ((chico_hue < 0.18) & (chico_hue > 0.05) &
                 (chico_sat > 0.55)  & (chico_sat < 0.80))
opened = area_opening(binarized_hsv, 5000)
ax[0].imshow(binarized_hsv)
ax[1].imshow(masked_image(img, opened))
plt.show()
# See? Now, that’s easier! This can be attributed to the Hue channel of the HSV color space
# that clearly identifies the objects on the images based on their hue.