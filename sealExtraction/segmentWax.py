import cv2 as cv
import imutils
import numpy as np

from utils import cropImageToContourAABB

#
#  SEGMENT WAX START

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

    return cropImageToContourAABB(image, contour)

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
