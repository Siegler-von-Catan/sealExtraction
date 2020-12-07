# Python 2/3 compatibility
from __future__ import print_function

import numpy as np

def getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels):
    maskRows = np.where(mask == 255)[0]

    return (len(maskRows) / numberOfWaxPixels)

def normalizeValues(nestedNpArray):
    highestValue = max({max(shapeValues) for (shapeValues) in nestedNpArray})
    lowestValue = min({min(shapeValues) for (shapeValues) in nestedNpArray})

    normalizedValues = list(map(lambda values: normalizeSimpleNumberArray(values, lowestValue, highestValue), nestedNpArray))
    return normalizedValues

def normalizeSimpleNumberArray(array, lowestValue, highestValue):
    return {(number-lowestValue) / (highestValue-lowestValue) for (number) in array}

