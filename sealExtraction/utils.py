import cv2 as cv
import imutils
import numpy as np

def getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels):
    maskRows = np.where(mask == 255)[0]

    return np.size(maskRows) / numberOfWaxPixels

def normalizeValues(nestedNpArray):
    highestValue = max({max(shapeValues) for shapeValues in nestedNpArray})
    lowestValue = min({min(shapeValues) for shapeValues in nestedNpArray})

    normalizedValues = list(map(lambda values: normalizeSimpleNumberArray(values, lowestValue, highestValue), nestedNpArray))
    return normalizedValues

def normalizeSimpleNumberArray(array, lowestValue, highestValue):
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
        cv.putText(image, "#{}".format(i + 1), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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