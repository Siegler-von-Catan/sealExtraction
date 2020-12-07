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

from sealExtraction import segmentWax, segmentMotive

inputPath = None

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
def main():
    jpgGlob = getJPGGlob()
        
    outputDirectory = inputPath + "edgeResult/"  
    os.makedirs(outputDirectory, exist_ok=True)
    for file in jpgGlob:
        image = cv.imread(file)
        result = segmentSeal(image)
        saveImageAsFile(result, file.split("/")[-1], outputDirectory)
        
    #handleKeyEvents()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

