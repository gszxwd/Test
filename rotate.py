__author__ = 'Xu'

#coding=utf-8
import cv2,os,sys,math
import numpy as np
from pylab import *
#matplotlib inline


def getBinary(path):
    im=cv2.imread(path,0)
    thresh,im=cv2.threshold(255-im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im

# shadow min or boundary box min???

def shadow(im,angel):
    x=[]
    height , width = im.shape
    for i in xrange(height):
        for j in xrange(width):
            if im[i][j]==255:
                x.append(j*math.cos(math.radians(angel)) - i*math.sin(math.radians(angel)))
    x=np.array(x)
    return x.max()-x.min()

def shadowTest():
    directory='weibo'
    pics=os.listdir(directory)
    pic = pics[0]
    im=getBinary(os.path.join(directory,pic))
    print shadow(im,30)

def getAngle(im):
    minShadow=500
    ans=0
    for angle in np.linspace(-89,90,180):
        thisShadow=shadow(im,angle)
        if minShadow>thisShadow:
            ans=angle
            minShadow=thisShadow
    return ans

def getAngleTest():
    directory='weibo'
    pics=os.listdir(directory)
    for pic in pics[:5]:
        im=getBinary(os.path.join(directory,pic))
        print getAngle(im)

def affine(im):
    height,width=im.shape
    angle=getAngle(im)
    pts1=np.float32([[width/2,height/2],[width/2,0],[0,height/2]])
    pts2=np.float32([[width/2,height/2],[width/2+height/2/math.tan(math.radians(angle)),0],[0,height/2]])
    M=cv2.getAffineTransform(pts1,pts2)
    dst=cv2.warpAffine(im,M,(width,height))
    return dst

# main
directory='weibo'
pics=os.listdir(directory)
for pic in pics:
    im = getBinary(os.path.join(directory,pic))
    height, width = im.shape
    angle = getAngle(im)
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    edge = max(height, width)
    dst = cv2.warpAffine(im, M, (edge, edge))
    newpic = 'rotate_'+pic
    cv2.imwrite(os.path.join(directory, newpic), dst)

# single image test
# gray()
# imshow(np.hstack([im,dst]))
pic = pics[6]
im = getBinary(os.path.join(directory,pic))
height, width = im.shape
angle = getAngle(im)
print angle
M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
edge = max(height, width)
dst = cv2.warpAffine(im, M, (edge, edge))
imshow(dst)
