# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:01:40 2021

@author: feyzi
"""
#%%

 # This function reads an image and returns its SIFT keypoints.
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
import matplotlib.pyplot as plt


img1 = cv2.imread('StarMap.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(gray1,None)

print(len(kp1),len(des1))
print(kp1,des1)

img2 = cv2.imread('Small_area.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp2,des2 = sift.detectAndCompute(gray2,None)

print(len(kp2),len(des2))

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:50], gray2, flags=2)
plt.imshow(img3),plt.show()
cv2.imwrite('Result_sift.png',img3)

print(len(matches))