#     SealExtraction - Extracting the motive out of stamps with Image Processing.
#     Copyright (C) 2021
#     Joana Bergsiek, Leonard Geier, Lisa Ihde, Tobias Markus, Dominik Meier, Paul Methfessel
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2 as cv
import imutils
import numpy as np

from utils import cropImageToContourAABB

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

    return result

def performGrabCutOn(image):
    mask = image.copy()
    mask_blur = cv.GaussianBlur(cv.cvtColor(mask, cv.COLOR_BGR2GRAY), (5,5), 0)
    init_mask = cv.threshold(mask_blur, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)[1]
    se = np.ones((3,3), np.uint8)
    init_mask = cv.morphologyEx(init_mask, cv.MORPH_OPEN,  se, iterations = 10)
    init_mask = cv.morphologyEx(init_mask, cv.MORPH_CLOSE, se, iterations = 10)
    init_mask[init_mask > 0] = cv.GC_PR_FGD
    init_mask[init_mask == 0] = cv.GC_PR_BGD
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    (mask, bgdModel, fgdModel) = cv.grabCut(image, init_mask, None, bgdModel,
        fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_MASK)

    return (mask == cv.GC_PR_FGD).astype("uint8") * 255

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
