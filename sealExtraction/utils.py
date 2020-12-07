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
   
def getRelativeMaskSizeToWaxSize(mask, numberOfWaxPixels):
    maskRows = np.where(mask == 255)[0]
    
    return (len(maskRows) / numberOfWaxPixels)

