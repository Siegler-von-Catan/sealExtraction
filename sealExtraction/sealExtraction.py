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

