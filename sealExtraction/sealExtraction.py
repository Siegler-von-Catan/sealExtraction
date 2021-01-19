#!/usr/bin/env python

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

from segmentMotive import segmentMotive
from segmentWax import segmentWax

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
    segmentedWax = segmentWax(image)
    segmentedMotive = segmentMotive(segmentedWax)
    result =  segmentedMotive
    cv.imwrite(args['output'], result)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

