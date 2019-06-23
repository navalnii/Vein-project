#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import cv2
from skimage import morphology

import os
from skimage.io import imread, imshow, imsave
import pickle
from matplotlib import pyplot as plt
from skimage.measure import regionprops

def connect_object(img, min_size, connectivity):
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def skeletonize_method(img, kernel):

    img = img.copy() 
    skel = img.copy()
    skel[:,:] = 0

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def get_kernel(size, types):
    
    if types == 'snowflake':
        
        kernel = np.zeros(size, np.uint8)
        s = kernel.shape[0]
        kernel[:,s//2] = 1
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if i == j:
                    kernel[i][j] = 1
                    kernel[i][kernel.shape[1]-1-j] = 1
                    
    elif types == 'vertical':
        
        kernel = np.zeros(size, np.uint8)
        s = kernel.shape[0]
        kernel[:, s//2] = 1
    
    elif types == 'lit_cross':
        
        kernel = np.zeros(size, np.uint8)
        s = kernel.shape[0]
        kernel[:, s//2] = 1
        kernel[s//2, s//2-1:s//2+2] = 1
    
    elif types == 'cross':
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    
    return kernel    



BASE_DIR 	= os.path.dirname(os.path.abspath(__file__))
image_dir 	= os.path.join(BASE_DIR, 'VPBase_corrected')

x_train = 	[]
labels 	= 	[]

check = True

# while check == True:
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('bmp', 'png')):

            path = os.path.join(root, file)
            img = cv2.imread(path,0)
            t, th3 = cv2.threshold(img,np.amin(img),np.amax(img),cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            region = regionprops(th3)
            minr, minc, maxr, maxc = region[0].bbox

            minr += 50
            minc += 50
            maxr -= 50
            maxc -= 50
            rected = img[minr:maxr, minc:maxc]

            norm_image = cv2.normalize(rected, None, alpha=0, beta=255,  
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            blur = cv2.blur(norm_image,(15,15),1)
            th = cv2.adaptiveThreshold(blur,np.amax(norm_image),cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,201,3)

            blur2 = cv2.blur(th,(28,28))
            sk = skeletonize_method(blur2, get_kernel((7, 7), 'cross'))
            connected = connect_object(sk,2500,8)
            blur3 = cv2.blur(connected,(10,10))
            resized = cv2.resize(blur3, (130, 100))
            out = np.uint8(resized)

            image_array = np.reshape(out,(1, -1))
            x_train.append(image_array[0])

        label = root.split('\\')[6]
        labels.append(label[2:])

with open('X.pickle', 'wb') as fl:
	pickle.dump(x_train, fl)


with open('y.pickle', 'wb') as fl:
	pickle.dump(labels, fl)
