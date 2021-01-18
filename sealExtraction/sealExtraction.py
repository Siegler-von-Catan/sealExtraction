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

from segmentMotive import segmentMotive
from segmentWax import segmentWax

def segmentSeal(image):
    """
    First, segment the wax from the image which contains the motive.
    Then, segment out the motive in the wax.
    """
    segmentedWax = segmentWax(image)
    segmentedMotive = segmentMotive(segmentedWax)
    return segmentedMotive

