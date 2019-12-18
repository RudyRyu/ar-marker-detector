import threading
import time
from threading import Lock

import cv2
import numpy as np

from common.utils import check_time

class Camera:
    def __init__(self, name, rtsp_link, map_positions, img_size=(1920, 1080),
                                                       calibration=True,
                                                       flip=None):

        self.name = name
        self.map_positions = map_positions
        self.last_map_pix = None
        self.calibration = calibration
        self.flip = flip
        self.img_size = img_size

        self.last_frame = None
        self.last_ready = None
        self.lock = Lock()

        self.record_list = []
        self.capture = cv2.VideoCapture(rtsp_link)

        cf_w = img_size[0] / 1920.
        cf_h = img_size[1] / 1080.

        K = np.array([[840.737*cf_w,        0.,           941.125*cf_w],
                      [0,             841.965*cf_h,       582.547*cf_h],
                      [0,                 0,                 1]])

        # Distortion Coefficients(kc) - 1st, 2nd
        d = np.array([-0.143462, 0.017334, 0.000636, 0.002194]) # just use first two terms

        cam_mat, roi = cv2.getOptimalNewCameraMatrix(K, d, img_size, 0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, d, None, cam_mat,
                                                           img_size, 5)

        thread = threading.Thread(target=self.rtsp_cam_buffer,
                                  args=(),
                                  name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = self.capture.read()

    @check_time
    def read_last_frame(self):
        if self.last_ready and (self.last_frame is not None):
            h, w = self.last_frame.shape[:2]
            if self.img_size != (w, h):
                img = cv2.resize(self.last_frame, self.img_size)
            else:
                img = self.last_frame

            if self.calibration:
                img = cv2.remap(self.last_frame, self.map1, self.map2,
                                cv2.INTER_CUBIC)

            if self.flip in [-1,0,1]:
                img = cv2.flip(img, self.flip)

            return img

        else:
            return None


    @check_time
    def read_each_frame(self):
        ret, img = self.capture.read()

        if ret:
            if self.img_size != (img.shape[1], img.shape[0]):
                img = cv2.resize(img, self.img_size)

            if self.calibration:
                img = cv2.remap(img, self.map1, self.map2, cv2.INTER_CUBIC)

            if self.flip in [-1,0,1]:
                img = cv2.flip(img, self.flip)

            return img

        else:
            return None
