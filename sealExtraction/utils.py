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

def getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels):
    maskRows = np.count_nonzero(mask == 255)

    return maskRows / numberOfWaxPixels

def normalizeValues(nestedNpArray):
    highestValue = max({max(shapeValues) for shapeValues in nestedNpArray})
    lowestValue = min({min(shapeValues) for shapeValues in nestedNpArray})

    normalizedValues = list(map(lambda values: normalizeSimpleNumberArray(values, lowestValue, highestValue), nestedNpArray))
    return normalizedValues

def normalizeSimpleNumberArray(array, lowestValue, highestValue):
    if lowestValue == highestValue:
        return {0 for _ in array}
    return {(number-lowestValue) / (highestValue-lowestValue) for number in array}


def findAndDrawContoursOn(thresh, image):
    contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    print("[INFO] {} unique contours found".format(len(contours)))

    drawGivenContoursOn(contours, image)

def drawGivenContoursOn(contours, image):
    for (i, c) in enumerate(contours):
        x,y,_w,_h = cv.boundingRect(c)
        cv.putText(image, "8.49", (int(x)-50, int(y)+50), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        cv.drawContours(image, [c], -1, (0, 255, 0), 6)


def cropImageToContourAABB(image, contour):
    # Adding borderPixels * 2 pixels for each dimension so that algorithms which create a shadow
    # like static saliency have a continous contour instead of split ones as
    # they would touch the image borders

    x,y,w,h = cv.boundingRect(contour)
    borderPixels = 20
    x = max(0, x - borderPixels)
    y = max(0, y - borderPixels)
    w = min(image.shape[1], w + 2 * borderPixels)
    h = min(image.shape[0], h + 2 * borderPixels)
    return image[y:y+h, x:x+w]