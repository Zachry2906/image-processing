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
        # cv.CV_64F digunakan untuk menentukan tipe data dari gambar, yaitu 64-bit float
        # ksize merupakan ukuran kernel yang digunakan
        # 1 dan 0 merupakan orde turunan pada sumbu x dan y
        sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        return cv.convertScaleAbs(cv.magnitude(sobel_x, sobel_y))
        # convertScaleAbs digunakan untuk mengkonversi array ke tipe data unsigned integer 8-bit
        # magnitude digunakan untuk menghitung magnitudo dari dua array
        # magnitudo merupakan akar kuadrat dari penjumlahan kuadrat dari dua array
    elif method == "Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        # kernelx merupakan kernel untuk mendeteksi tepi pada sumbu x, yang berisi array 3x3
        # 1,1,1 digunakan untuk mendeteksi tepi pada bagian atas gambar
        # 0,0,0 digunakan untuk mengabaikan bagian tengah gambar
        # -1,-1,-1 digunakan untuk mendeteksi tepi pada bagian bawah gambar
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        # kernely merupakan kernel untuk mendeteksi tepi pada sumbu y, yang berisi array 3x3
        # -1,0,1 digunakan untuk mendeteksi tepi pada bagian kiri gambar
        # -1,0,1 digunakan untuk mengabaikan bagian tengah gambar
        # -1,0,1 digunakan untuk mendeteksi tepi pada bagian kanan gambar
        img_prewittx = cv.filter2D(img, -1, kernelx)
        # filter2D digunakan untuk melakukan konvolusi pada gambar, yang menerima tiga parameter, gambar, tipe data, dan kernel
        # -1 digunakan untuk menentukan tipe data dari gambar, yaitu sama dengan gambar input
        img_prewitty = cv.filter2D(img, -1, kernely)
        # filter2D digunakan untuk melakukan konvolusi pada gambar, yang menerima tiga parameter, gambar, tipe data, dan kernel
        # -1 digunakan untuk menentukan tipe data dari gambar, yaitu sama dengan gambar input
        return cv.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
        # addWeighted digunakan untuk menggabungkan dua gambar dengan bobot tertentu, yang menerima lima parameter, dua gambar, bobot gambar pertama, bobot gambar kedua, dan konstanta
        # 0.5 dan 0.5 digunakan untuk memberikan bobot yang sama pada kedua gambar
    elif method == "Robert":
        roberts_cross_v = np.array([[1, 0], [0, -1]])
        # roberts_cross_v merupakan kernel untuk mendeteksi tepi pada sumbu vertikal, yang berisi array 2x2
        # 1,0 digunakan untuk mendeteksi tepi pada bagian atas gambar
        # 0,-1 digunakan untuk mendeteksi tepi pada bagian bawah gambar
        roberts_cross_h = np.array([[0, 1], [-1, 0]])
        # roberts_cross_h merupakan kernel untuk mendeteksi tepi pada sumbu horizontal, yang berisi array 2x2
        # 0,1 digunakan untuk mendeteksi tepi pada bagian kiri gambar
        # -1,0 digunakan untuk mendeteksi tepi pada bagian kanan gambar
        vertical = ndimage.convolve(img, roberts_cross_v)
        # convolve digunakan untuk melakukan konvolusi pada gambar, yang menerima dua parameter, gambar dan kernel
        horizontal = ndimage.convolve(img, roberts_cross_h)
        # convolve digunakan untuk melakukan konvolusi pada gambar, yang menerima dua parameter, gambar dan kernel
        
        #tidak dipakai
        #return np.sqrt(np.square(horizontal) + np.square(vertical)).astype(np.uint8)
        # sqrt digunakan untuk menghitung akar kuadrat dari array input
        
        # Gabungkan tepi horizontal dan vertikal
        edge_magnitude = np.sqrt(np.square(horizontal) + np.square(vertical))

        # Menormalkan besarnya tepi ke kisaran 0-255
        edge_magnitude_normalized = np.uint8(255 * (edge_magnitude - np.min(edge_magnitude)) / (np.max(edge_magnitude) - np.min(edge_magnitude)))

        return edge_magnitude_normalized

# Negative
def convert_to_negative(img):
    return cv.bitwise_not(img)
# bitwise_not digunakan untuk melakukan operasi bitwise NOT pada gambar
# NOT merupakan operasi yang mengubah nilai piksel menjadi nilai yang berlawanan
# contoh: 0 -> 255, 255 -> 0

# Binary
def convert_to_binary(img, threshold):
    _, binary = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    # threshold digunakan untuk mengubah gambar menjadi gambar biner
    # threshold menerima empat parameter, gambar, nilai threshold, nilai maksimum, dan metode threshold
    # cv.THRESH_BINARY merupakan metode threshold yang digunakan untuk mengubah nilai piksel yang lebih besar dari nilai threshold menjadi nilai maksimum
    # _ digunakan untuk menyimpan nilai threshold yang digunakan
    # metode threshold ynag lain adalah cv.THRESH_BINARY_INV, cv.THRESH_TRUNC, cv.THRESH_TOZERO, dan cv.THRESH_TOZERO_INV
    return binary

# Smooth
def convert_to_smooth(img, factor):
    return cv.GaussianBlur(img, (factor, factor), 0)
# GaussianBlur digunakan untuk menghaluskan gambar dengan kernel Gaussian 2D
# GaussianBlur menerima tiga parameter, gambar, ukuran kernel, dan sigmaX
# sigmaX merupakan deviasi standar pada sumbu x

# Brightness
def change_brightness(img, factor):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # cvtColor digunakan untuk mengubah warna gambar ke HSV
    # COLOR_BGR2HSV digunakan untuk mengubah warna gambar dari BGR ke HSV
    # HSV adalah singkatan dari Hue, Saturation, dan Value
    # harus diubah ke HSV agar dapat mengubah nilai kecerahan gambar
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    # np.clip digunakan untuk membatasi nilai piksel antara 0 dan 255
    # hsv[:,:,2] digunakan untuk mengakses komponen Value dari gambar
    # factor digunakan untuk mengubah nilai kecerahan gambar
    # cp.clip menerima tiga parameter, array, nilai minimum, dan nilai maksimum
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # cvtColor digunakan untuk mengubah warna gambar dari HSV ke BGR
    #cvtColor menerima dua parameter, gambar dan metode yang digunakan

# Equalization
def equalization(img):
    return cv.equalizeHist(img)
# equalizeHist digunakan untuk menyeimbangkan histogram gambar dengan metode equalization
# equalizeHist menerima satu parameter, gambar yang akan diubah histogramnya

# Rotate
def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)
# rotate digunakan untuk memutar gambar sebesar angle derajat
# rotate menerima tiga parameter, gambar, sudut rotasi, dan reshape
# reshape=False digunakan untuk mempertahankan ukuran gambar yang sama setelah rotasi

# Flip
def flip(img, arrow):
    if arrow == "Horizontal":
        return cv.flip(img, 1)
    # flip digunakan untuk membalikkan gambar, 1 untuk flip horizontal, 0 untuk flip vertical, dan -1 untuk flip horizontal dan vertical
    elif arrow == "Vertical":
        return cv.flip(img, 0)
    elif arrow == "Both":
        return cv.flip(img, -1)

# Contrast
def contrast(img, factor):
    return np.clip((img - 128) * factor + 128, 0, 255).astype(np.uint8)
# np.clip digunakan untuk membatasi nilai piksel antara 0 dan 255
# img - 128 digunakan untuk mengurangi nilai piksel dengan 128
# factor digunakan untuk mengubah nilai kontras gambar
# 128 digunakan untuk menambahkan nilai piksel dengan 128
# astype digunakan untuk mengubah tipe data gambar menjadi unsigned integer 8-bit
# harus diubah ke unsigned integer 8-bit agar nilai piksel berada di antara 0 dan 255

# Sharpness
def sharpness(img, factor):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # kernel merupakan kernel sharpening yang digunakan untuk meningkatkan ketajaman gambar
    # kernel berisi array 3x3
    # -1,-1,-1 digunakan untuk mengurangi nilai piksel di sekitar piksel pusat
    # 9 digunakan untuk menambahkan nilai piksel pusat
    sharpened = cv.filter2D(img, -1, kernel)
    # filter2D digunakan untuk melakukan konvolusi pada gambar, yang menerima tiga parameter, gambar, tipe data, dan kernel
    # -1 digunakan untuk menentukan tipe data dari gambar, yaitu sama dengan gambar input
    return cv.addWeighted(img, 1-factor, sharpened, factor, 0)
    # addWeighted digunakan untuk menggabungkan dua gambar dengan bobot tertentu, yang menerima lima parameter, dua gambar, bobot gambar pertama, bobot gambar kedua, dan konstanta
    # 1-factor digunakan untuk memberikan bobot gambar asli yang lebih besar
    # factor digunakan untuk memberikan bobot gambar yang sudah di sharpening