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

from glob import glob

import argparse
import sys
import os

import cv2 as cv

from sealExtraction import segmentSeal

inputPath = None

def initializArgumentParser():
    """
    Initialize documentation for the command line arguments and return the arguments extracted
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True,
	help="path to directory containing jpg pictures to process")
    return ap

def getJPGGlob():
    """
    Return the glob (https://docs.python.org/3/library/glob.html) of all containing jpg files
    in the given images directory. Closes program if no glob could be created by eg. not
    finding the given path
    """
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
    outputPath = directoryToSave + imageName
    cv.imwrite(outputPath, image)


def main():
    """
    Read in all jpg images which lie in IMAGES_DIRECTORY_PATH,
    Extract the white background,
    Normalize image size,
    Extract Motive,
    Save each result as folder in given path
    """
    jpgGlob = getJPGGlob()

    outputDirectory = inputPath + "edgeResult/"
    os.makedirs(outputDirectory, exist_ok=True)
    for file in jpgGlob:
        image = cv.imread(file)
        result = segmentSeal(image)
        saveImageAsFile(result, file.split("/")[-1], outputDirectory)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

