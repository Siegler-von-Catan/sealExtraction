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

inputPath = None

tenChooseTwoCache = [
    [1, 0],[2, 0],[2, 1],[3, 0],[3, 1],
    [3, 2],[4, 0],[4, 1],[4, 2],[4, 3],
    [5, 0],[5, 1],[5, 2],[5, 3],[5, 4],
    [6, 0],[6, 1],[6, 2],[6, 3],[6, 4],
    [6, 5],[7, 0],[7, 1],[7, 2],[7, 3],
    [7, 4],[7, 5],[7, 6],[8, 0],[8, 1],
    [8, 2],[8, 3],[8, 4],[8, 5],[8, 6],
    [8, 7],[9, 0],[9, 1],[9, 2],[9, 3],
    [9, 4],[9, 5],[9, 6],[9, 7],[9, 8],
    ]

'''
Keeps a key listener running and reacts to down presses of keys.
See program documentation for the key mapping.
'''
def handleKeyEvents():
    while True:
        ch = cv.waitKey()
        if ch == 27:
            break

            
'''
Initialize documentation for the command line arguments and return the arguments extracted
'''
def initializArgumentParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True,
	help="path to directory containing jpg pictures to process")
    return ap

'''
Return the glob (https://docs.python.org/3/library/glob.html) of all containing jpg files
in the given images directory. Closes program if no glob could be created by eg. not
finding the given path
'''
def getJPGGlob():
    global inputPath
    ap = initializArgumentParser()
    args = vars(ap.parse_args())
    inputPath = args["directory"]
    imageRegex = inputPath + "/*.jpg"
    jpgGlob = glob(imageRegex)
    
    if jpgGlob is None: 
        print("Failed to load images:", inputPath)
        sys.exit(1)
    
    return jpgGlob
    
def saveImageAsFile(image, imageName, directoryToSave):
    outputPath = outputDirectory + imageName 
    cv.imwrite(outputPath, image)
    
'''
First, segment the wax from the image which contains the motive.
Then, segment out the motive in the wax.
'''    
def segmentSeal(image):
    segmentedWax = segmentWax(image)
    segmentedMotive = segmentMotive(segmentedWax)
    return segmentedMotive


'-------------------------------------------------SEGMENT WAX START-----------------------------------------------------'

'''
Perform a grabcut (https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html) with the whole image as the 
foreground frame to extract the wax.
It may be that the grabcut will fail to filter out the entire background. To still get a similiar result like we expect to, 
we OTSU threshold the image, filter out noise and boolean merge it with the failed grabcut attempt mask. 
We perform a boolean merge to refine the contours of the wax to get a more precise result with less white borders.
We also delete all areas of the picture which are not in the Wax's AABB to reduce image size for later operations.
'''
def segmentWax(image):
    resizedImage = imutils.resize(image.copy(), width=500)
    foreGroundMask = performGrabCutOn(resizedImage)
    
    # In case the grabcut fails to delete the white background
    gray = imutils.resize(cv.cvtColor(image, cv.COLOR_BGR2GRAY), width=500)
    whiteMask = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    whiteMask = cv.morphologyEx(whiteMask, cv.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations = 25)
    combinedMask = cv.bitwise_and(foreGroundMask,foreGroundMask,mask = whiteMask)
    
    combinedMask = cv.resize(combinedMask,  (image.shape[1], image.shape[0]))
    filterOutSmallObjectAreas(combinedMask)
    
    result = cv.bitwise_and(image,image,mask = combinedMask)
    #result = cropImageToWaxAABB(result, combinedMask)
    
    return result    

def performGrabCutOn(image):
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    height, width, _ = image.shape
    rect = (10,10,width-1,height-50)
    (mask, bgdModel, fgdModel) = cv.grabCut(image, mask, rect, bgdModel,
	fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)
    
    foreGroundMask = (mask == cv.GC_PR_FGD).astype("uint8") * 255
    
    return foreGroundMask

def cropImageToWaxAABB(image, waxMask):
    contours = cv.findContours(waxMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contours)[0]
    x,y,w,h = cv.boundingRect(contour)
    # Adding borderPixels * 2 pixels for each dimension so that algorithms which create a shadow
    # like static saliency have a continous contour instead of split ones as
    # they would touch the image borders
    
    #So....actually cropping it makes it worse for saliency? OK?
    borderPixels = 20;
    x = max(0, x - borderPixels)
    y = max(0, y - borderPixels)
    w = min(image.shape[1], w + 2 * borderPixels)
    h = min(image.shape[0], h + 2 * borderPixels)
    return image[y:y+h, x:x+w]

'''
Delete any contour that is not the biggest one in the given threshold binary image
'''
def filterOutSmallObjectAreas(thresh):
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for i in range(1, len(contours)):
        cv.fillPoly(thresh, pts =[contours[i]], color=(0,0,0)) 
        
'-------------------------------------------------SEGMENT WAX END-------------------------------------------------------'      

'----------------------------------------------SEGMENT MOTIVE START-----------------------------------------------------'

'''
Perform a static saliency (https://docs.opencv.org/master/d8/d65/group__saliency.html) of the image and create a black-white
threshold image. With that mask we can pair contours, fit an ellipse through them and choose the one with the highest probabilty
of containing the motive. That fitted elipse is then choosen as the motive.
'''
def segmentMotive(waxSegmentedImage):
    waxThreshold = cv.threshold( cv.cvtColor(waxSegmentedImage, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)[1]
    saliencyMask = getSaliencyMask(waxSegmentedImage)
    referenceImage = getReferenceImageFor(waxSegmentedImage)
    
    # obvious todo here
    contours = cv.findContours(saliencyMask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    
    saliencyMask = cropImageToContoursAABB(saliencyMask, contours)
    waxSegmentedImage = cropImageToContoursAABB(waxSegmentedImage, contours)
    referenceImage = cropImageToContoursAABB(referenceImage, contours)
    
    contours = cv.findContours(saliencyMask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    # todo end
    
    waxThreshold = cv.threshold( cv.cvtColor(waxSegmentedImage, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)[1]
    numberOfWaxPixels = np.where(waxThreshold > 0)[0]
    mask = getMostLikelyMaskFor(contours, tenChooseTwoCache, referenceImage, len(numberOfWaxPixels))
    maskContours = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    maskContours = imutils.grab_contours(maskContours)
    
    drawGivenContoursOn(maskContours, waxSegmentedImage)
    
    return waxSegmentedImage

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

def cropImageToContoursAABB(image, contours):
    points = np.concatenate((contours[0], contours[1], contours[2], contours[3], contours[4], contours[5], contours[6], contours[7], contours[8], contours[9]), axis=0)
    x,y,w,h = cv.boundingRect(points)
    borderPixels = 20;
    x = max(0, x - borderPixels)
    y = max(0, y - borderPixels)
    w = min(image.shape[1], w + 2 * borderPixels)
    h = min(image.shape[0], h + 2 * borderPixels)
    return image[y:y+h, x:x+w]

def getReferenceImageFor(waxSegmentedImage):
    waxSegmentedImage = cv.GaussianBlur(waxSegmentedImage, (21, 21), 0)
    sobel = getSobelOf(waxSegmentedImage)
    return sobel

def getSobelOf(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv.cv.CV_32F if imutils.is_cv2() else cv.CV_32F
    gradX = cv.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)
    
    thresh = cv.threshold(gradient,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return thresh

def getMostLikelyMaskFor(allContours, contoursPairs, referenceImage, numberOfWaxPixels):
    pairMasks = [getMasksForContourPair(allContours[pair[0]], allContours[pair[1]], referenceImage, numberOfWaxPixels) for pair in contoursPairs]
    pairMasks = list(filter(lambda masks: len(masks) > 0, pairMasks))
    
    scoreSums = getCriteriaWeightedScores(pairMasks, referenceImage, numberOfWaxPixels)
    
    #drawPairMasksAndScoresOn(pairMasks[:5], scoreSums[:5], original)
    
    return getMaskWithHighestScore(scoreSums, pairMasks)
    
def drawPairMasksAndScoresOn(pairMasks, scoreSums, imageToDrawOn):
    for (index, shapeMasks) in enumerate(pairMasks):
        drawMasksAndScoresOn(shapeMasks, scoreSums[index], imageToDrawOn)
        
def drawMasksAndScoresOn(shapeMasks, scoreSums, imageToDrawOn):
    for (index, shapeMask) in enumerate(shapeMasks):
        drawMaskAndSumOnImage(shapeMask, scoreSums[index], imageToDrawOn)

def drawMaskAndSumOnImage(shapeMask, scoreSum, imageToDrawOn):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)
    
    for (i, c) in enumerate(shapeContours):
        x,y,w,h = cv.boundingRect(c)
        cv.putText(imageToDrawOn, "{}".format(scoreSum), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.drawContours(imageToDrawOn, [c], -1, (0, 255, 0), 6)   
    
def getMaskWithHighestScore(scoreSums, pairMasks):
    highestValueInPairValues = list(map(lambda values: max(values), scoreSums))
    highestScore = max(highestValueInPairValues)
    highestScoreIndex = highestValueInPairValues.index(highestScore)
    belongingMasks = pairMasks[highestScoreIndex]
    maxValues = scoreSums[highestScoreIndex]
    belongingMask = maxValues.index(max(maxValues))
    return belongingMasks[belongingMask]
    
def getCriteriaWeightedScores(pairMasks, referenceImage, numberOfWaxPixels):
    pairDensities = [getDensitiesForPairMasks(masks, referenceImage) for masks in pairMasks]
    pairDensities = normalizeValues(pairDensities)
    
    pairDistances = [getDistancesForPairMasks(masks, referenceImage) for masks in pairMasks]
    pairDistances = normalizeValues(pairDistances)
    pairDistances = list(map(lambda values: list(map(lambda value: 1 - value, values)), pairDistances))
    
    pairAngles = [getAngleDifferencesForPairMasks(masks) for masks in pairMasks]
    pairAngles = normalizeValues(pairAngles)
    pairAngles = list(map(lambda values: list(map(lambda value: 1 - value, values)), pairAngles))
    
    pairSizeDistributions = [getDistributionValuesForPairMasks(masks, numberOfWaxPixels) for masks in pairMasks]
    pairSizeDistributions = normalizeValues(pairSizeDistributions)
    
    return [getSingleWeightedScores(pairDensities[i], pairDistances[i], pairAngles[i], pairSizeDistributions[i]) for i, pair in enumerate(pairMasks)]

def getSingleWeightedScores(densityScoresForPair, distanceScoresForPair, angleScoresForPair, distributionScoresForPair): 
    return [ getWeightedScore(density, distance, angle, distribution) for density, distance, angle, distribution in zip(densityScoresForPair, distanceScoresForPair, angleScoresForPair, distributionScoresForPair)]

def getWeightedScore(densityScore, distanceScore, angleScore, distributionScore):
    return densityScore + 3*distanceScore + 2*angleScore + 2.4*distributionScore

def getMasksForContourPair(contourA, contourB, referenceImage, numberOfWaxPixels):
    'Create basic shapes to test how likely they contain the motive'
    masks = getShapeMasksForContourPair(contourA, contourB, referenceImage)
    
    filteredMasks = list(filter(lambda mask: not shouldBeFilteredOut(mask, referenceImage, numberOfWaxPixels), masks))
    return filteredMasks

def shouldBeFilteredOut(mask, referenceImage, numberOfWaxPixels):
    if (isEmptyMask(mask)): return True
    if (getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels) <= 0.15): return True
    if (boundingBoxIsTouchingImageBorder(mask, referenceImage)): return True
    return False
    
def isEmptyMask(mask):
    return not np.any(mask > 0)

def boundingBoxIsTouchingImageBorder(shapeMask, referenceImage):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)
    
    imageHeight, imageWidth = referenceImage.shape
    x,y,w,h = cv.boundingRect(shapeContours[0])
    if x == 0 or y == 0 or x+w >= imageWidth-1 or y+h >= imageHeight-1: return True
    
    return False
    
def getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels):
    maskRows = np.where(mask == 255)[0]
    
    return (len(maskRows) / numberOfWaxPixels)

def getShapeMasksForContourPair(contourA, contourB, referenceImage):
    points = np.concatenate((contourA, contourB))
    ellipse = cv.fitEllipse(points)
    
    (x,y),radius = cv.minEnclosingCircle(points)
    center = (int(x),int(y))
    radius = int(radius)
    
    rect = cv.minAreaRect(points)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    
    ellipseMask = np.zeros(referenceImage.shape[:2],np.uint8)
    circleMask = np.zeros(referenceImage.shape[:2],np.uint8)
    boxMask = np.zeros(referenceImage.shape[:2],np.uint8)
    
    cv.ellipse(ellipseMask,ellipse,255,-1)
    cv.circle(circleMask,center,radius,255,-1)
    cv.drawContours(boxMask,[box],0,255,-1)
    
    return [ellipseMask, circleMask, boxMask]

def normalizeValues(nestedNpArray):
    highestValue = max({max(shapeValues) for (shapeValues) in nestedNpArray})
    lowestValue = min({min(shapeValues) for (shapeValues) in nestedNpArray})
    
    normalizedValues = list(map(lambda values: normalizeSimpleNumberArray(values, lowestValue, highestValue), nestedNpArray))
    return normalizedValues

def normalizeSimpleNumberArray(array, lowestValue, highestValue):
    return {(number-lowestValue) / (highestValue-lowestValue) for (number) in array}

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

'-----------------------------------------------SCORING END-------------------------------------------------------------'
    
'-----------------------------------------------SEGMENT MOTIVE END------------------------------------------------------'

        
def findAndDrawContoursOn(thresh, image):
    contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    print("[INFO] {} unique contours found".format(len(contours)))
    
    drawGivenContoursOn(contours, image)

def drawGivenContoursOn(contours, image):
    for (i, c) in enumerate(contours):
        x,y,w,h = cv.boundingRect(c)
        cv.putText(image, "#{}".format(i + 1), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.drawContours(image, [c], -1, (0, 255, 0), 6)    


'''
Read in all jpg images which lie in IMAGES_DIRECTORY_PATH,
Extract the white background, 
Normalize image size,
Extract Motive,
Save each result as folder in given path
'''
if __name__ == '__main__':
    jpgGlob = getJPGGlob()
        
    outputDirectory = inputPath + "edgeResult/"  
    os.makedirs(outputDirectory, exist_ok=True)
    for file in jpgGlob:
        image = cv.imread(file)
        result = segmentSeal(image)
        saveImageAsFile(result, file.split("/")[-1], outputDirectory)
        
    #handleKeyEvents()
    cv.destroyAllWindows()