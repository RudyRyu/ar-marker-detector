from __future__ import print_function
from __future__ import division
import numpy as np

try:
    import cv2
except ImportError:
    raise Exception('Error: OpenCv is not installed')

from numpy import array, rot90

from ar_marker_detector.coding import decode, extract_hamming_code
from ar_marker_detector.marker import MARKER_SIZE, HammingMarker
from common.utils import check_time


BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]


def validate_and_turn(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')
    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3
    marker = rot90(marker, k=rotation)
    return marker

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

@check_time
def detect_markers(img, scale=1):
    """
    This is the main function for detecting markers in an image.

    Input:
      img: a color or grayscale image that may or may not contain a marker.

    Output:
      a list of found markers. If no markers are found, then it is an empty list.
    """

    img= cv2.add(img.copy(), np.array([30.0]))
    if len(img.shape) > 2:
        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width = img.shape
        gray = img
    markers_list = []
    rect_list = []
    scaled = cv2.resize(gray, (int(width*scale), int(height*scale)), cv2.INTER_CUBIC)
    # gray = cv2.GaussianBlur(gray, (3, 3), 1.1)
    _, scaled = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(scaled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    # We only keep the long enough contours
    min_contour_length = min(width, height) / 50
    contours = [contour for contour in contours if len(contour) > min_contour_length]
    warped_size = 70
    canonical_marker_coords = array(
        (
            (0, 0),
            (warped_size - 1, 0),
            (warped_size - 1, warped_size - 1),
            (0, warped_size - 1)
        ),
        dtype='float32')


    for contour in contours:
        con_area = cv2.contourArea(contour)
        if con_area <= 1000 or con_area >= 2500:
            continue

        cnt_len = cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, cnt_len * 0.02, True)
        if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve) and cv2.contourArea(approx_curve) > 100):
            continue
        cnt = approx_curve.reshape(-1, 2)
        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
        if max_cos >= 0.3:
            continue
        approx_curve = np.array(approx_curve)
        approx_curve = (approx_curve / scale).astype(np.int32)
        rect_list.append(approx_curve)
        sorted_curve = array(
            cv2.convexHull(approx_curve, clockwise=False),
            dtype='float32'
        )
        persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
        warped_img = cv2.warpPerspective(gray, persp_transf, (warped_size, warped_size))

        # kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        # sharpen_warped_img = cv2.filter2D(warped_img,-1,kernel_sharpen_1)

        # _, warped_bin = cv2.threshold(warped_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, warped_bin = cv2.threshold(warped_img, 165, 255, cv2.THRESH_BINARY)

        marker = warped_bin.reshape(
            [MARKER_SIZE, warped_size // MARKER_SIZE, MARKER_SIZE, warped_size // MARKER_SIZE]
        )
        marker = marker.mean(axis=3).mean(axis=1)
        marker[marker < 127] = 0
        marker[marker >= 127] = 1

        try:
            marker = validate_and_turn(marker)
            hamming_code = extract_hamming_code(marker)
            marker_id = int(decode(hamming_code), 2)
            markers_list.append(HammingMarker(id=marker_id, contours=approx_curve.astype('int')))

        except ValueError as e:
            # print(e)

            # print(marker_id)

            # cv2.imshow('warped_bin', warped_bin)
            # cv2.imshow('warped_img', warped_img)
            # cv2.waitKey()
            continue

    return markers_list, rect_list
