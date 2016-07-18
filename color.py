__author__ = 'Xu'

import cv2
import numpy as np

# testing 237,247,252/253
#filename = '1/1_253.jpg'
def color_seg(filename):
    # segment the background
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    val, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    img_new = img*(img2/255)

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

    # segment symbols according the distance
    def get_yslices(bins):
        ret = []
        x = y = -1
        for i in range(len(bins)):
            if x==-1 and bins[i] > 0:
                x = i-1
            if x!=-1 and y==-1 and bins[i]<1e-10:
                y = i
                ret.append([x,y])
                x = y = -1
        return ret

    # k-means segmentation
    yslices = get_yslices(np.array(ybins)[:,0])
    samples = []
    for i in range(len(yslices)):
        samples = samples + range(yslices[i][0], yslices[i][1])

    # define customized distance
    def distance(v1, v2, i=0, j=0):
        weights = 50
        ret = sum(np.power(v1-v2, 2)) + weights*((i-j)**2)
        return ret

    # use the matrix to store the cluster info, first column save belongings, second column save distance
    clusters = np.mat(np.ones((len(samples),2)))*-1

    # step1: init centroids (TODO: improve, e.g. use filtered color info, but which color?)
    # method 1 (use symbol width info)
    #width_min = 8
    width_max = 28
    longbins = []
    removebins = []
    interval = 2
    # handle cases which bins are close
    for i in range(1,len(yslices)):
        if (yslices[i][0]-yslices[i-1][1]) <= interval:
            removebins.append(i)
    removebins.reverse()
    for i in removebins:
        yslices[i-1][1] = yslices[i][1]
        yslices.remove(yslices[i])

    # handle long bin case
    removebins = []
    for i in range(len(yslices)):
        if (yslices[i][1] - yslices[i][0]) > width_max:
            longbins.append(i)
    for i in longbins:
        a = yslices[i][0]
        b = yslices[i][1]
        removebins.append(i)
        d = (b-a)/width_max
        for j in range(d+1):
            yslices.append([round((j/float(d+1))*(b-a)+a), round(((j+1)/float(d+1))*(b-a)+a)])
    removebins.reverse()
    for i in removebins:
        yslices.remove(yslices[i])
    yslices.sort()

    # use the center of each bin as centroids
    k = len(yslices)
    centroids = []
    for i in range(k):
        centroids.append(int((yslices[i][0]+yslices[i][1])/2))
    centroids = np.array(centroids)

    # method 2 (using percentile number)
    # k = 7
    # centroids = []
    # for i in range(k):
    #     centroids.append(np.int(np.percentile(samples, (100)*(i+1)/(k+1))))
    # centroids = np.array(centroids)

    # method 3 (using distance segmentation result)
    # k = 7
    # centroids = []
    # if len(yslices) >= k:
    #     for i in range(k):
    #         centroids.append((yslices[i][0]+yslices[i][1])/2)
    #     centroids = np.array(centroids)
    # else:
    #     centroids = np.random.choice(samples, k, False)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # for each sample
        for i in range(len(samples)):
            minDist = 99999
            index = 0
            # step2: find the centroid which is closest
            for j in range(len(centroids)):
                dist = distance(np.array(ybins[samples[i]]), np.array(ybins[centroids[j]]), samples[i], centroids[j])
                if dist < minDist:
                    minDist = dist
                    index = j
            # step3: update its cluster info
            if clusters[i,0] != index:
                clusterChanged = True
                clusters[i,:] = index, minDist
        # step4: update centroids
        for j in range(len(centroids)):
            tmp = np.nonzero(clusters[:,0]==j)[0]
            pointsInCluster = np.array(samples)[tmp]
            centroids[j] = np.int(np.mean(pointsInCluster))

    # get the new segmented slices
    seg = clusters[:,0].tolist()
    j = 0
    flag = True
    start = end = 0
    yslices_new = []
    for i in range(len(seg)):
        if (seg[i][0] == j) and (not flag):
            flag = True
            end = samples[i-1]
            yslices_new.append([start, end])
        if (seg[i][0] == j) and flag:
            flag = False
            start = samples[i]
            #print i, start
            j = j+1
        if i == len(seg)-1:
            end = samples[i]
            yslices_new.append([start, end])

    # get the segmented images
    def get_xslices(xbins):
        x = y = -1
        for i in range(len(xbins)):
            if xbins[i] > 0:
                x = i-1
                break
        for i in range(len(xbins)-1, -1, -1):
            if xbins[i] > 0:
                y = i+1
                break
        return [x,y]

    def get_seg_images(img, yslices):
        imgs = []
        # region [x1, x2, y1, y2] list saved for further processing
        regions = []
        for i in range(len(yslices)):
            img2 = img[:,yslices[i][0]:yslices[i][1]]
            nrow = img2.shape[0]
            xbins = []
            for j in range(nrow):
                count = np.count_nonzero(img2[j, :])
                xbins.append(count)
            xslices = get_xslices(xbins)
            img2 = img2[xslices[0]:xslices[1],:]
            imgs.append(img2)
            regions.append([xslices[0],xslices[1],yslices[i][0],yslices[i][1]])
        return imgs, regions

    imgs, regions = get_seg_images(thresh, yslices_new)

    # save the segmented images
    for i in range(len(imgs)):
        new_filename = filename.split(".")[0]+"-part-"+str(i+1)+"."+filename.split(".")[1]
        cv2.imwrite(new_filename, 255-imgs[i])

import os
dir = "3/"
filelist = os.listdir(dir)
for filename in filelist:
    color_seg(dir+filename)


# test color distance function 1
# img_part1 = img_new[33:44,15:30,:]
# r1 = np.mean(img_part1[(img_part1[:,:,0] > 0)][:,0])
# g1 = np.mean(img_part1[(img_part1[:,:,1] > 0)][:,1])
# b1 = np.mean(img_part1[(img_part1[:,:,2] > 0)][:,2])
# color1 = np.array([r1, g1, b1])
#
# img_part2 = img_new[11:33,34:56,:]
# r2 = np.mean(img_part2[(img_part2[:,:,0] > 0)][:,0])
# g2 = np.mean(img_part2[(img_part2[:,:,1] > 0)][:,1])
# b2 = np.mean(img_part2[(img_part2[:,:,2] > 0)][:,2])
# color2 = np.array([r2, g2, b2])
#
# def ncc(c1, c2):
#     c1 = np.array(c1)
#     c2 = np.array(c2)
#     return c1.dot(c2)/((sum(c1**2)**0.5)*(sum(c2**2)**0.5))
#
# def ssd(c1, c2):
#     return sum((np.array(c1)-np.array(c2))**2)

# test color distance 2
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
# convert to hsu
# hsu = []
# for y in ybins:
#     rgb = sRGBColor(y[0],y[1],y[2],True)
#     print y
#     lab = convert_color(rgb, LabColor)
#     hsu.append([lab.lab_l, lab.lab_a, lab.lab_b])
# plot(hsu)

from colormath.color_diff import delta_e_cie1994
# ssd_bins = []
# deltae_bins = []
# for i in range(1, ncol-1):
#     v1 = ssd(ybins[i], ybins[i+1]) + ssd(ybins[i], ybins[i-1])
#     ssd_bins.append(v1)
#     rgb1 = sRGBColor(ybins[i][0],ybins[i][1],ybins[i][2],True)
#     rgb2 = sRGBColor(ybins[i+1][0],ybins[i+1][1],ybins[i+1][2],True)
#     rgb3 = sRGBColor(ybins[i-1][0],ybins[i-1][1],ybins[i-1][2],True)
#     lab1 = convert_color(rgb1, LabColor)
#     lab2 = convert_color(rgb2, LabColor)
#     lab3 = convert_color(rgb3, LabColor)
#     v2 = delta_e_cie1994(lab1, lab2) + delta_e_cie1994(lab1, lab3)
#     deltae_bins.append(v2)

