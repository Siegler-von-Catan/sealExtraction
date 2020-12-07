# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import imutils
import numpy as np
from ensure import ensure

from saliency import getSaliencyMask
from scoring import getCriteriaWeightedScores, getAngleDifferenceToEvenRotation, getDensitiesForPairMasks, getDistancesForPairMasks, getAngleDifferencesForPairMasks, getDistributionValuesForPairMasks, getSymmetryValuesForPairMaks

from utils import getRelativeMaskSizeToWaxSize, normalizeValues

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

def segmentSeal(image):
    """
    First, segment the wax from the image which contains the motive.
    Then, segment out the motive in the wax.
    """
    segmentedWax = segmentWax(image)
    segmentedMotive = segmentMotive(segmentedWax)
    return segmentedMotive


# SEGMENT WAX START

def segmentWax(image):
    """
    Perform a grabcut (https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html) with the whole image as the
    foreground frame to extract the wax.
    It may be that the grabcut will fail to filter out the entire background.
    To still get a similiar result like we expect to, we OTSU threshold the image,
    filter out noise and boolean merge it with the failed grabcut attempt mask.
    We perform a boolean merge to refine the contours of the wax to get a more precise result with less white borders.
    We also delete all areas of the picture which are not in the Wax's AABB to reduce image size for later operations.
    """
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

def filterOutSmallObjectAreas(thresh):
    """
    Delete any contour that is not the biggest one in the given threshold binary image
    """
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for i in range(1, len(contours)):
        cv.fillPoly(thresh, pts =[contours[i]], color=(0,0,0))

# SEGMENT WAX END

# SEGMENT MOTIVE START

def segmentMotive(waxSegmentedImage):
    """
    Perform a static saliency (https://docs.opencv.org/master/d8/d65/group__saliency.html) of the image and create a black-white
    threshold image. With that mask we can pair contours, fit an ellipse through them and choose the one with the highest probabilty
    of containing the motive. That fitted elipse is then choosen as the motive.
    """
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
    mask = getMostLikelyMaskFor(contours, tenChooseTwoCache, referenceImage, len(numberOfWaxPixels), waxSegmentedImage)
    maskContours = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    maskContours = imutils.grab_contours(maskContours)

    drawGivenContoursOn(maskContours, waxSegmentedImage)

    return waxSegmentedImage

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

def getMostLikelyMaskFor(allContours, contoursPairs, referenceImage, numberOfWaxPixels, original):
    pairMasks = [getMasksForContourPair(allContours[pair[0]], allContours[pair[1]], referenceImage, numberOfWaxPixels) for pair in contoursPairs]
    pairMasks = list(filter(lambda masks: len(masks) > 0, pairMasks))

    scoreSums = getCriteriaWeightedScores(pairMasks, referenceImage, numberOfWaxPixels)

    drawPairMasksAndScoresOn(pairMasks, scoreSums, original)

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

def getMasksForContourPair(contourA, contourB, referenceImage, numberOfWaxPixels):
    'Create basic shapes to test how likely they contain the motive'
    masks = getShapeMasksForContourPair(contourA, contourB, referenceImage)

    filteredMasks = list(filter(lambda mask: not shouldBeFilteredOut(mask, referenceImage, numberOfWaxPixels), masks))
    return filteredMasks

def shouldBeFilteredOut(mask, referenceImage, numberOfWaxPixels):
    if isEmptyMask(mask):
        return True
    if getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels) <= 0.15:
        return True
    if boundingBoxIsTouchingImageBorder(mask, referenceImage):
        return True
    if getAngleDifferenceToEvenRotation(mask) >= 25:
        return True
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

# SEGMENT MOTIVE END
