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

from utils import getRelativeMaskSizeToWaxSize

'-------------------------------------------------------SCORING START----------------------------------------------------'


def getDensitiesForPairMasks(pairMasks, referenceImage):
    return list(map(lambda mask: getDensityOfMaskedArea(mask, referenceImage), pairMasks))

def getDensityOfMaskedArea(mask, referenceImage):
    maskCopy = imutils.resize(mask.copy(), width=500)
    referenceCopy = imutils.resize(referenceImage.copy(), width=500)
    
    rows, cols = np.where(maskCopy == 255)
    
    pixelsInMask = referenceCopy[rows, cols]
    whitePixels = np.where(pixelsInMask == 255)
    if(len(whitePixels[0]) == 0): return 0
    return len(rows)/len(whitePixels[0])


def getDistancesForPairMasks(pairMasks, referenceImage):
    return list(map(lambda shapeMask: getShapeCenterToImageCenterDistance(shapeMask, referenceImage), pairMasks))

def getShapeCenterToImageCenterDistance(shapeMask, referenceImage):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)

    # FIXME Is this correct?
    image = referenceImage
    
    # compute the center of the contour
    shapeMoment = cv.moments(shapeContours[0])
    shapeCenterX = int(shapeMoment["m10"] / shapeMoment["m00"])
    shapeCenterY = int(shapeMoment["m01"] / shapeMoment["m00"])
    
    imageHeight, imageWidth = image.shape[:2]
    imageCenter = np.array([imageWidth/2, imageHeight/2])
    pairCenter = np.array([shapeCenterX, shapeCenterY])
    
    distance = np.linalg.norm(imageCenter-pairCenter)
    return distance
    
def getAngleDifferencesForPairMasks(pairMasks):
    return list(map(lambda mask: getAngleDifferenceToEvenRotation(mask), pairMasks))
    
def getAngleDifferenceToEvenRotation(shapeMask):
    shapeAngle = abs(getAngleFromShapeMask(shapeMask))
    # even rotations are 0, 90, 180, 270, 360. as the difference of 45 to the next one is equal to 135, module 90 here
    shapeAngle = shapeAngle % 90
    
    if (shapeAngle <= 45): return shapeAngle
    else: return 90 - shapeAngle
    
def getAngleFromShapeMask(shapeMask):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)
    
    rect = cv.minAreaRect(shapeContours[0])
    return rect[2]

def getDistributionValuesForPairMasks(pairMasks, numberOfWaxPixels):
    return list(map(lambda mask: getDistributionValueOfRelativeShapeSize(mask, numberOfWaxPixels), pairMasks))

def getDistributionValueOfRelativeShapeSize(shapeMask, numberOfWaxPixels):
    return normpdf(getRelativeMaskSizeToWaxSize(shapeMask, numberOfWaxPixels), 0.6, 0.15)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def getSymmetryValuesForPairMaks(pairMasks, referenceImage):
    return list(map(lambda mask: getSymmetryValueForMask(mask, referenceImage), pairMasks))
    
def getSymmetryValueForMask(shapeMask, referenceImage):
    # Our approach is to first get the moments of the contour the mask describes.
    # With the contour's center of gravity and orientation we fit a line through it,
    # drawing it black to separate the mask into two even shapes in which we can
    # individually check how many pixels in the reference image are white in them.
    # Finally, we compare the count and return <todo>
    maskCopy = imutils.resize(shapeMask.copy(), width=700)
    referenceCopy = imutils.resize(referenceImage.copy(), width=700)
    
    shapeHalvesMask, (startX, startY), (endX, endY) = splitShapeMaskInEvenHalves(maskCopy)
    
    shapeHalvesContours = cv.findContours(shapeHalvesMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeHalvesContours = imutils.grab_contours(shapeHalvesContours)
    ensure(len(shapeHalvesContours) == 2).equals(True)
    
    oneHalfMask = np.zeros_like(shapeHalvesMask)
    cv.drawContours(oneHalfMask, shapeHalvesContours, 0, (255, 255, 255), -1)
    
    rows, cols = np.where(oneHalfMask == 255)
    allSymmetryEqualityTuples = [pointEqualsMirrorPointInReferenceImage(x, y, startX, startY, endX, endY, referenceCopy) for x,y in zip(rows, cols)]
    mirroredPointEqualsCount = len(np.where(allSymmetryEqualityTuples)[0])
    print(mirroredPointEqualsCount / len(rows))
    
    cv.drawContours(referenceCopy, shapeHalvesContours, -1, (255, 255, 255), 4)
    cv.imshow("symmetry", referenceCopy)
    
    cv.waitKey(0)
    return mirroredPointEqualsCount / len(rows)
    
def pointEqualsMirrorPointInReferenceImage(pointX, pointY, startX, startY, endX, endY, referenceImage):
    mirroredPointAlongLine = getMirroredPointAlong(pointX, pointY, startX, startY, endX, endY)
    pointValueInReferenceImage = referenceImage[pointX, pointY]
    mirroredPointValueInReferenceImage = referenceImage[min(mirroredPointAlongLine[0], referenceImage.shape[0]-1), min(mirroredPointAlongLine[1], referenceImage.shape[1]-1)]
    return pointValueInReferenceImage == mirroredPointValueInReferenceImage
    
def getMirroredPointAlong(pointX, pointY, startX, startY, endX, endY):
    # Set line to equation Ax + By + C = 0
    A = endY - startY
    B = -(endX - startX)
    C = -A * startX - B * startY
    
    M = math.sqrt(A * A + B * B)
    AN = A / M
    BN = B / M
    CN = C / M
    
    D = AN * pointX + BN * pointY + CN
    
    PMX = pointX - 2 * AN * D
    PMY = pointY - 2 * BN * D
    
    return (int(PMX), int(PMY))

def splitShapeMaskInEvenHalves(shapeMask):
    maskCopy = shapeMask.copy()
    shapeContours = cv.findContours(maskCopy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)
    
    moments = cv.moments(shapeContours[0])
    
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    #theta = 27
    theta = 0.5 * np.arctan2(2*moments["mu11"], moments["mu20"] - moments["mu02"])
    
    lineLength = max(maskCopy.shape)
    startX = int(-lineLength * np.cos(theta) + center[0])
    startY = int(-lineLength * np.sin(theta) + center[1])
    endX = int(lineLength * np.cos(theta) + center[0])
    endY = int(lineLength * np.sin(theta) + center[1])
    
    #startX = int( center[0])
    #startY = int(-lineLength * np.sin(theta) + center[1])
    #endX = int( center[0])
    #endY = int(lineLength * np.sin(theta) + center[1])
    
    cv.line(maskCopy, (startX, startY), (endX, endY), (0, 0, 0), 5)
    
    return (maskCopy, (startX, startY), (endX, endY))
    

'-----------------------------------------------SCORING END-------------------------------------------------------------'
    
'-----------------------------------------------SEGMENT MOTIVE END------------------------------------------------------'

