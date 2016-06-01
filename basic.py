__author__ = "Xu"

import cv2
import numpy as np

def basic_seg(filename):
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    val, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    ncol = thresh.shape[1]
    ybins = []
    for i in range(ncol):
        count = np.count_nonzero(thresh[:,i])
        ybins.append(count)

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

    yslices = get_yslices(ybins)

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

    imgs, regions = get_seg_images(thresh, yslices)

    # save the segmented images
    # for i in range(len(imgs)):
    #     new_filename = filename.split(".")[0]+"-part-"+str(i+1)+"."+filename.split(".")[1]
    #     cv2.imwrite(new_filename, 255-imgs[i])

    # save the regions
    f = open("region.txt", "a")
    f.writelines(str(regions)+"\n")
    f.close()

import os
dir = "1/"
filelist = os.listdir(dir)
for filename in filelist:
    basic_seg(dir+filename)
