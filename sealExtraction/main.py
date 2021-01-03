#!/usr/bin/env python

'''
Detecting seals' motives for Paul Arnold Grun's Data set (https://codingdavinci.de/de/daten/siegelsammlung-paul-arnold-grun).

We do not care about the writing, background or the edges of the pressed down wax. Thus the extracted result will only contain the shape and motive
of the used stamp.

Usage:
  sealExtraction.py [-h] --o OUTPUT INPUT

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
    ap.add_argument("-o", "--output", required=True, help='output file')
    ap.add_argument('input', metavar="INPUT", help='input file')
    return ap


def main():
    """
    Read in jpg image,
    Extract the white background,
    Extract Motive,
    Save each result as folder in given path
    """

    ap = initializArgumentParser()
    args = vars(ap.parse_args())

    image = cv.imread(args['input'])
    result = segmentSeal(image)
    cv.imwrite(args['output'], result)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

