__author__ = 'Xu'

import cv2
import numpy as np

region = [[33, 44, 15, 30], [11, 33, 34, 56], [29, 48, 77, 95], [19, 34, 106, 115], [17, 36, 136, 155], [18, 35, 167, 191], [26, 29, 192, 195]]

filename = '1/1_229.jpg'
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
val, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
img_new = img*(img2/255)

# test the color distance function
img_part1 = img_new[33:44,15:30,:]
r1 = np.mean(img_part1[(img_part1[:,:,0] > 0)][:,0])
g1 = np.mean(img_part1[(img_part1[:,:,1] > 0)][:,1])
b1 = np.mean(img_part1[(img_part1[:,:,2] > 0)][:,2])
color1 = np.array([r1, g1, b1])

img_part2 = img_new[11:33,34:56,:]
r2 = np.mean(img_part2[(img_part2[:,:,0] > 0)][:,0])
g2 = np.mean(img_part2[(img_part2[:,:,1] > 0)][:,1])
b2 = np.mean(img_part2[(img_part2[:,:,2] > 0)][:,2])
color2 = np.array([r2, g2, b2])

def ncc(c1, c2):
    return c1.dot(c2)/((sum(c1**2)**0.5)*(sum(c2**2)**0.5))

def ssd(c1, c2):
    return sum((np.array(c1)-np.array(c2))**2)

# add color info to hist
def color_mean(x):
    y = [0]
    for i in x:
        if i==0:
            continue
        y.append(i)
    return np.mean(y)

ncol = thresh.shape[1]
ybins = []
imgr = img_new[:,:,0]
imgg = img_new[:,:,1]
imgb = img_new[:,:,2]
for i in range(ncol):
    r = color_mean(imgr[:,i])
    g = color_mean(imgg[:,i])
    b = color_mean(imgb[:,i])
    ybins.append([r,g,b])

color_bins = []
for i in range(ncol-1):
    color = ssd(ybins[i], ybins[i+1])
    color_bins.append(color)

