from scipy import ndimage
import numpy as np
import cv2
import os
import sys
#ORIGINAL IMAGE
gambar = str(sys.argv[1])
read_gambar = os.open(gambar, os.O_RDONLY)
img1 = cv2.imread(gambar)
img2 = cv2.resize(img1, (0, 0), fx=0.6, fy=0.6)
#Data Image
originY, originX = 0, 0 # initialize the origin
dimensions = img2.shape # get dimensions of image
height, width = img2.shape[:2] # get the image height and width 
size = img2.size #ukuran data pada media penyimpan
dtype = img2.dtype #image datatype (kedalaman bit)
# compute the center of the image
# which is simply the width and height
# divided by two
(centerX, centerY) = (width // 2, height // 2)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#SMOOTHING WITH GAUSSIAN FILTER
img = gray.astype(np.float64)
img_blur =cv2.GaussianBlur(img, (5,5),0)
#BLACK AND WHITE
#Memberikan Gradient Magnitude
def sobel_filters(img):
 Kx = np.array([[-1, 0, 1], 
 [-2, 0, 2], 
 [-1, 0, 1]], np.float32)
 Ky = np.array([[1, 2, 1],
[0, 0, 0], 
 [-1, -2, -1]], np.float32)
 
 Ix = ndimage.filters.convolve(img, Kx)
 Iy = ndimage.filters.convolve(img, Ky)
 
 G = np.hypot(Ix, Iy)
 G = G / G.max() * 255
 theta = np.arctan2(Iy, Ix) 
 return (G, theta)
#Menampilkan Gradient Magnitude
img_sobel, arah = sobel_filters(img_blur)
#Non-Maximum Supperssion
def non_max_suppression(img, D):
 M, N = img.shape
 Z = np.zeros((M,N), dtype=np.int32)
 angle = D * 180. / np.pi
 angle[angle < 0] += 180
 
 for i in range(1,M-1):
 for j in range(1,N-1):
 try:
 q = 255
 r = 255
 
 #angle 0
 if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
 q = img[i, j+1]
 r = img[i, j-1]
 #angle 45
 elif (22.5 <= angle[i,j] < 67.5):
 q = img[i+1, j-1]
 r = img[i-1, j+1]
 #angle 90
 elif (67.5 <= angle[i,j] < 112.5):
 q = img[i+1, j]
 r = img[i-1, j]
 #angle 135
elif (112.5 <= angle[i,j] < 157.5):
 q = img[i-1, j-1]
 r = img[i+1, j+1]
 if (img[i,j] >= q) and (img[i,j] >= r):
 Z[i,j] = img[i,j]
 else:
 Z[i,j] = 0
 except IndexError as e:
 pass
 
 return Z
non_max_s = non_max_suppression(img_sobel, arah)
#Threshold
def threshold(img):
 highThreshold= 40
 lowThreshold = 20
 M, N = img.shape
 res = np.zeros((M,N), dtype=np.int32)
 
 weak = np.int32(25)
 strong = np.int32(255)
 
 strong_i, strong_j = np.where(img >= highThreshold)
 zeros_i, zeros_j = np.where(img < lowThreshold)
 
 weak_i, weak_j = np.where((img <= highThreshold) & (img >= 
lowThreshold))
 
 res[strong_i, strong_j] = strong
 res[weak_i, weak_j] = weak
 
 return res
th = threshold(non_max_s)
#EDGE DETECTION BY HYSTERESIS
def hysteresis(img):
 M, N = img.shape 
 strong = 255
 weak = 25 
 for i in range(1, M-1):
 for j in range(1, N-1):
 if (img[i,j] == weak):
 try:
 if ((img[i+1, j-1] == strong) or 
 (img[i+1, j] == strong) or 
 (img[i+1, j+1] == strong) or 
 (img[i, j-1] == strong) or 
 (img[i, j+1] == strong) or 
 (img[i-1, j-1] == strong) or 
 (img[i-1, j] == strong) or 
 (img[i-1, j+1] == strong)):
 img[i, j] = strong
 else:
 img[i, j] = 0
 except IndexError as e:
 pass
 return img
if _name_ == "_main_":
from PIL import Image, ImageDraw
input_image = Image.open('kepalaR.jpg')
output_image = Image.new("RGB", input_image.size)
final = hysteresis(th)
# print("Hasil Edge Detection by Hysteresis")
cv2.imwrite("canny.jpg",final)
#PROCESSING
img = cv2.imread('canny.jpg')
output = img.copy()
#Convert image to Grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Blur image
gray = cv2.GaussianBlur(gray, (3,3), 0)
#Canny edge detection
edges = cv2.Canny(gray, 50, 200)
#detect circles
circles = cv2.HoughCircles(image=gray,
 method=cv2.HOUGH_GRADIENT, 
 dp=1.1, 
 param1=40,
 param2=50, 
 minDist=1000, 
 minRadius=0, 
 maxRadius=0)
rcircles = np.uint16(np.around(circles))
if circles is not None:
 circles = np.round(circles[0, :]).astype("int")
 # print("Number of circles:", len(circles))
 # Count circles
 count=1 
 for (x, y, r) in circles:
 # Create outer circle
 cv2.circle(output, (x,y), r, (0, 0, 255), 3)
 # Create center rectangle
 cv2.rectangle(output, (x-2, y-2), (x+2, y+2), (0,255,0), -1)
 # Add radius to center
 cv2.putText(output, str(r), 
 (x-15, y-5), 
 cv2.FONT_HERSHEY_COMPLEX_SMALL, 
 0.7, (0, 0, 0), 1)
cv2.waitKey(0)
count += 1
#Radius pixels to cm
px = 0.026458333333333
cm = r * px
import math
#Head Circumference
pi = 3.14
circumference = 2 * pi * cm
print((round(circumference,2)),'cm')
# return circumference
