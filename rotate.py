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

def shadow(im,angel):
    x=[]
    height , width = im.shape
    for i in xrange(height):
        for j in xrange(width):
            if im[i][j]==255:
                x.append(j-1.0*i/math.tan(math.radians(angel)))
    x=np.array(x)
    return x.max()-x.min()

def shadowTest():
    directory='weibo'
    pics=os.listdir(directory)
    for pic in pics[:10]:
        im=getBinary(os.path.join(directory,pic))
        print shadow(im,50)
        figure()
        gray()
        imshow(im)

def getAngle(im):
    minShadow=500
    ans=90
    for angle in np.linspace(45,135,91):
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
        figure()
        imshow(im)

def affine(im):
    height,width=im.shape
    angle=getAngle(im)
    pts1=np.float32([[width/2,height/2],[width/2,0],[0,height/2]])
    pts2=np.float32([[width/2,height/2],[width/2+height/2/math.tan(math.radians(angle)),0],[0,height/2]])
    M=cv2.getAffineTransform(pts1,pts2)
    dst=cv2.warpAffine(im,M,(width,height))
    return dst

def affineTest():
    directory='weibo'
    pics=os.listdir(directory)
    for pic in pics[:1]:
        im=getBinary(os.path.join(directory,pic))
        dst=affine(im)
        gray()
        figure()
        imshow(np.hstack([im,dst]))
        axis('off')

affineTest()