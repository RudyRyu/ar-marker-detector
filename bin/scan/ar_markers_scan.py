#!/usr/bin/env python

from __future__ import print_function
import argparse
import cv2
from common.cam_reader import Camera
from ar_marker_detector.detect import detect_markers

parser = argparse.ArgumentParser()
parser.add_argument('-cam')
parser.add_argument('-frame_size', type=int, nargs='+')

if __name__ == '__main__':
    print('Press "q" to quit')
    args = parser.parse_args()
    capture = Camera(args.cam, tuple(args.frame_size))
    view_size = (1280, 720)

    while True:
        frame = capture.read()
        if frame is not None:
            markers = detect_markers(frame)
            for marker in markers:
                marker.highlite_marker(frame)
            resized = cv2.resize(frame, view_size)
            cv2.imshow('Result', resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
