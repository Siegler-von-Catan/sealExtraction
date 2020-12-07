#!/usr/bin/env python

'''
Detecting seals' motives for Paul Arnold Grun's Data set (https://codingdavinci.de/de/daten/siegelsammlung-paul-arnold-grun).

We do not care about the writing, background or the edges of the pressed down wax. Thus the extracted result will only contain the shape and motive
of the used stamp.

Usage:
  sealExtraction.py [-h] --d IMAGES_DIRECTORY_PATH

Keys:
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
from glob import glob

import math
import cv2 as cv
import imutils
import argparse
import sys
import numpy as np
import os
import itertools
from ensure import ensure

'-----------------------------------------------------SALIENCY START-----------------------------------------------------'


def getSaliencyMask(waxSegmentedImage):
    waxSegmentedImageCopy = waxSegmentedImage.copy()
    waxSegmentedImageCopy = imutils.resize(waxSegmentedImageCopy, width=900)
    waxSegmentedImageCopy = preProcessForSaliency(waxSegmentedImageCopy)
    
    threshMap =  getSaliencyThreshMap(waxSegmentedImageCopy)
    threshMap = postProcessThreshMask(threshMap)
    
    threshMap = cv.resize(threshMap,  (waxSegmentedImage.shape[1], waxSegmentedImage.shape[0]))
        
    return threshMap
 
def preProcessForSaliency(waxSegmentedImage):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    waxSegmentedImage = cv.filter2D(waxSegmentedImage,-1,filter)
    waxSegmentedImage = cv.GaussianBlur(waxSegmentedImage, (13, 13), 0)
    return waxSegmentedImage

def getSaliencyThreshMap(image):
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv.threshold(saliencyMap,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return threshMap

def postProcessThreshMask(threshMap):
    contours = cv.findContours(threshMap.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    splittingIndex = 10
    interestingContours = contours[:splittingIndex]
     # the first two contours are always from the wax and the shadow generated by saliency so leave them out...
    interestingContours.pop(0)
    interestingContours.pop(1)
    # ...and delete the shadow by filling the borders between the first two contours black
    threshMap = deleteShadow(threshMap, contours[1])
    
    deleteUninterestingContours(contours, threshMap, splittingIndex)
    
    return threshMap


'''
Delete all contours of the mask which are beyong the splittingIndex of the given list containing all contours
'''
def deleteUninterestingContours(allContours, maskToDrawOn, splittingIndex):
    uninterestingContours = allContours[splittingIndex:]
    for (i, c) in enumerate(uninterestingContours):
        cv.fillPoly(maskToDrawOn, pts =[c], color=(0,0,0)) 
        
'''
Static saliency creates a shadow around the object which becomes part of the mask we want to create.
Delete it by overdrawing it
'''
def deleteShadow(saliencyTreshMap, innerShadowContour):
    shadowMask = np.zeros(saliencyTreshMap.shape[:2],np.uint8)
    cv.fillPoly(shadowMask, pts =[innerShadowContour], color=(255,255,255)) 
    shadowMask = cv.bitwise_and(saliencyTreshMap,saliencyTreshMap,mask = shadowMask)
    
    # In some cases, there is no shadow (if the mask is also touching image borders)
    # Check and return the orignal just in case
    contours = cv.findContours(shadowMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if (len(contours) == 1):
        return saliencyTreshMap
    else: return shadowMask
        

'-------------------------------------------------------SALIENCY END----------------------------------------------------'

