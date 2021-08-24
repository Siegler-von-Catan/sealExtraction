import cv2 as cv
from sealExtraction.segmentMotive import segmentMotive
from sealExtraction.segmentWax import segmentWax

def test_combined_segment_wax_and_segment_motive():
    image = cv.imread("test_images/test_01.jpg")
    tmp = segmentWax(image)
    tmp = segmentMotive(tmp)

    # This is just a lazy check if the result looks somehow okay
    assert tmp.shape == (2472, 1036, 4)


def test_combined_only_segment_motive():
    image = cv.imread("test_images/test_01.jpg")
    tmp = segmentMotive(image)

    # This is just a lazy check if the result looks somehow okay
    assert tmp.shape == (2472, 1036, 4)
