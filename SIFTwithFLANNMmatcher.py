# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:04:11 2021

@author: feyzi
"""

#%%

 # This function reads images and returns theirs SIFT keypoints.
 # Then it finds the croppped image by comparing these keys points.
 #   Input parameters:
 #    imageFile: the file name for the image.

 #   Returned:
 #     image: the image array in double format
 #     descriptors: a K-by-128 matrix, where each row gives an invariant
 #         descriptor for one of the K keypoints.  The descriptor is a vector
 #         of 128 values normalized to unit length.



import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img2 = cv2.imread('StarMap.png',0)          # queryImage
img1 = cv2.imread('Small_area.png',0)       # trainImage


sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

M, mask = cv2.findHomography(pts1,pts2, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)


  
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
cv2.imwrite('Result3_sift.png',img3)
