import cv2 as cv
import numpy as np
from skimage import filters
from scipy import ndimage
from PIL import Image, ImageEnhance

# Grayscale
def convert_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect edge
def detect_edge(img, method):
    if method == "Canny":
        return cv.Canny(img, 100, 200)
    elif method == "Sobel":
        sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        return cv.convertScaleAbs(cv.magnitude(sobel_x, sobel_y))
    elif method == "Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv.filter2D(img, -1, kernelx)
        img_prewitty = cv.filter2D(img, -1, kernely)
        return cv.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
    elif method == "Robert":
        roberts_cross_v = np.array([[1, 0], [0, -1]])
        roberts_cross_h = np.array([[0, 1], [-1, 0]])
        vertical = ndimage.convolve(img, roberts_cross_v)
        horizontal = ndimage.convolve(img, roberts_cross_h)
        return np.sqrt(np.square(horizontal) + np.square(vertical)).astype(np.uint8)

# Negative
def convert_to_negative(img):
    return cv.bitwise_not(img)

# Binary
def convert_to_binary(img, threshold):
    _, binary = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return binary

# Smooth
def convert_to_smooth(img, factor):
    return cv.GaussianBlur(img, (factor, factor), 0)

# Brightness
def change_brightness(img, factor):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# Equalization
def equalization(img):
    return cv.equalizeHist(img)

# Rotate
def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

# Flip
def flip(img, arrow):
    if arrow == "Horizontal":
        return cv.flip(img, 1)
    elif arrow == "Vertical":
        return cv.flip(img, 0)
    elif arrow == "Both":
        return cv.flip(img, -1)

# Contrast
def contrast(img, factor):
    return np.clip((img - 128) * factor + 128, 0, 255).astype(np.uint8)

# Sharpness
def sharpness(img, factor):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv.filter2D(img, -1, kernel)
    return cv.addWeighted(img, 1-factor, sharpened, factor, 0)