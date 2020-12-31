import math
import cv2 as cv
import imutils
import numpy as np
from ensure import ensure
from scipy import stats

from utils import getRelativeMaskSizeToWaxSize, normalizeValues

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

    pairSymmetries = [getSymmetryValuesForPairMasks(masks, referenceImage) for masks in pairMasks]
    pairSymmetries = normalizeValues(pairSymmetries)
    pairSymmetries = list(map(lambda values: list(map(lambda value: 1 - value, values)), pairSymmetries))

    return [getSingleWeightedScores(pairDensities[i], pairDistances[i], pairAngles[i], pairSizeDistributions[i], pairSymmetries[i]) for i, pair in enumerate(pairMasks)]

def getSingleWeightedScores(densityScoresForPair, distanceScoresForPair, angleScoresForPair, distributionScoresForPair, symmetryForPair):
    return [ getWeightedScore(density, distance, angle, distribution, symmetry) for density, distance, angle, distribution, symmetry in zip(densityScoresForPair, distanceScoresForPair, angleScoresForPair, distributionScoresForPair, symmetryForPair)]

def getWeightedScore(densityScore, distanceScore, angleScore, distributionScore, symmetry):
    return densityScore + 3*distanceScore + 2*angleScore + 2.5*distributionScore + 2*symmetry

def getDensitiesForPairMasks(pairMasks, referenceImage):
    return list(map(lambda mask: getDensityOfMaskedArea(mask, referenceImage), pairMasks))

def getDensityOfMaskedArea(mask, referenceImage):
    maskCopy = imutils.resize(mask.copy(), width=500)
    referenceCopy = imutils.resize(referenceImage.copy(), width=500)

    rows, cols = np.where(maskCopy == 255)

    pixelsInMask = referenceCopy[rows, cols]
    whitePixels = np.where(pixelsInMask == 255)
    if len(whitePixels[0]) == 0:
        return 0
    return len(rows)/len(whitePixels[0])

def getDistancesForPairMasks(pairMasks, referenceImage):
    return list(map(lambda shapeMask: getShapeCenterToImageCenterDistance(shapeMask, referenceImage), pairMasks))

def getShapeCenterToImageCenterDistance(shapeMask, referenceImage):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)

    shapeMoment = cv.moments(shapeContours[0])
    shapeCenterX = int(shapeMoment["m10"] / shapeMoment["m00"])
    shapeCenterY = int(shapeMoment["m01"] / shapeMoment["m00"])

    imageHeight, imageWidth = referenceImage.shape[:2]
    imageCenter = np.array([imageWidth/2, imageHeight/2])
    pairCenter = np.array([shapeCenterX, shapeCenterY])

    return np.linalg.norm(imageCenter-pairCenter)

def getAngleDifferencesForPairMasks(pairMasks):
    return list(map(lambda mask: getAngleDifferenceToEvenRotation(mask), pairMasks))

def getAngleDifferenceToEvenRotation(shapeMask):
    shapeAngle = abs(getAngleFromShapeMask(shapeMask))
    # even rotations are 0, 90, 180, 270, 360. as the difference of 45 to the next one is equal to 135, module 90 here
    shapeAngle = shapeAngle % 90

    if shapeAngle <= 45:
        return shapeAngle
    else:
        return 90 - shapeAngle

def getAngleFromShapeMask(shapeMask):
    shapeContours = cv.findContours(shapeMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeContours = imutils.grab_contours(shapeContours)
    ensure(len(shapeContours) == 1).equals(True)

    rect = cv.minAreaRect(shapeContours[0])
    return rect[2]

def getDistributionValuesForPairMasks(pairMasks, numberOfWaxPixels):
    return list(map(lambda mask: getDistributionValueOfRelativeShapeSize(mask, numberOfWaxPixels), pairMasks))

def getDistributionValueOfRelativeShapeSize(shapeMask, numberOfWaxPixels):
    return stats.norm.pdf(getRelativeMaskSizeToWaxSize(shapeMask, numberOfWaxPixels), loc=0.6, scale=0.15) + stats.norm.pdf(getRelativeMaskSizeToWaxSize(shapeMask, numberOfWaxPixels), loc=0.98, scale=0.01)

def getSymmetryValuesForPairMasks(pairMasks, referenceImage):
    return list(map(lambda mask: getSymmetryValueForMask(mask, referenceImage), pairMasks))

def getSymmetryValueForMask(shapeMask, referenceImage):
    # Our approach is to first get the moments of the contour the mask describes.
    # With the contour's center of gravity and orientation we fit a line through it,
    # drawing it black to separate the mask into two even shapes. For one of those shapes
    # (oneHalfMask) we iterate through every (white) pixel p to count how many white pixels n
    # are in p's neighborhood in the referenceImage. We then count the mirrored pixel's p'
    # count n' and check the difference between n and n'. Doing that, we do not compare "exact"
    # symmetry by checking referenceImage(p) == referenceImage(p'), as this resulted in unreliable
    # overall values for a shape due to the high count of black pixels in the referenceImage.
    # We get the differences abs(n-n') 0 meaning they have the same count
    # of white pixels in their neighborhood and are seen as totally symmetric.
    # To get an overall score for a shape we calculate the L2-Norm of all the differences.

    maskCopy = imutils.resize(shapeMask.copy(), width=500)
    referenceCopy = imutils.resize(referenceImage.copy(), width=500)

    shapeHalvesMask, (startX, startY), (endX, endY) = splitShapeMaskInEvenHalves(maskCopy)

    shapeHalvesContours = cv.findContours(shapeHalvesMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shapeHalvesContours = imutils.grab_contours(shapeHalvesContours)
    ensure(len(shapeHalvesContours) == 2).equals(True)

    oneHalfMask = np.zeros_like(shapeHalvesMask)
    cv.drawContours(oneHalfMask, shapeHalvesContours, 0, (255, 255, 255), -1)

    rows, cols = np.where(oneHalfMask == 255)
    mirroredPointsDifferenceCounts = [getWhitePixelDifferenceToMirroredPoint(x, y, startX, startY, endX, endY, 2, referenceCopy) for x,y in zip(rows, cols)]
    totalCount = np.linalg.norm(mirroredPointsDifferenceCounts)
    return totalCount

def getWhitePixelDifferenceToMirroredPoint(pointX, pointY, startX, startY, endX, endY, kernelRadius, referenceImage):
    ownWhitePixelCount = getNumberOfWhitePixelsInNeighborhood(pointX, pointY, kernelRadius, referenceImage)

    mirroredPointAlongLine = getMirroredPointAlong(pointX, pointY, startX, startY, endX, endY)
    mirroredWhitePixelCount =  getNumberOfWhitePixelsInNeighborhood(mirroredPointAlongLine[0], mirroredPointAlongLine[1], kernelRadius, referenceImage)

    whitePixelDifferenceCount = abs(ownWhitePixelCount-mirroredWhitePixelCount)

    return whitePixelDifferenceCount

def getNumberOfWhitePixelsInNeighborhood(pointX, pointY, kernelRadius, referenceImage):
    (imageHeight, imageWidth) = referenceImage.shape[:2]

    whitePixelsCount = cv.countNonZero(referenceImage[max(pointX-kernelRadius, 0):min(pointX+kernelRadius+1, imageWidth),
                   max(pointY-kernelRadius,0):min(pointY+kernelRadius+1, imageHeight)].flatten())

    return whitePixelsCount

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
    theta = 0.5 * np.arctan2(2*moments["mu11"], moments["mu20"] - moments["mu02"])

    lineLength = max(maskCopy.shape)
    startX = int(-lineLength * np.cos(theta) + center[0])
    startY = int(-lineLength * np.sin(theta) + center[1])
    endX = int(lineLength * np.cos(theta) + center[0])
    endY = int(lineLength * np.sin(theta) + center[1])

    cv.line(maskCopy, (startX, startY), (endX, endY), (0, 0, 0), 5)

    return (maskCopy, (startX, startY), (endX, endY))
