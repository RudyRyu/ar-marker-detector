import threading
import time
from threading import Lock

import cv2
import numpy as np

from common.utils import check_time

class Camera:
    def __init__(self, name, rtsp_link, map_positions, last_map_pix, img_size):

        self.name = name
        self.map_positions = map_positions
        self.last_map_pix = last_map_pix

        self.last_frame = None
        self.last_ready = None
        self.lock = Lock()

        K = np.array([[9784.521+232.291, 0., 959.784],
                      [0, 9432.565+225.503, 758.442],
                      [0, 0, 1]])

        # Distortion Coefficients(kc) - 1st, 2nd
        # just use first two terms
        d = np.array([-16.271484, 279.268252, -0.134332, 0.149862, 0])
        cam_mat, roi = cv2.getOptimalNewCameraMatrix(K, d, img_size, 0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, d, None, cam_mat,
                                                           img_size, 5)

        self.capture = cv2.VideoCapture(rtsp_link)

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
    def read_each_img(self):
        ret, img = self.capture.read()
        if ret:
            return cv2.remap(img, self.map1, self.map2, cv2.INTER_CUBIC)
        else:
            return None

    @check_time
    def read_last_frame(self, distortion=False):
        if self.last_ready and self.last_frame is not None:
            if not distortion:
                return self.last_frame
            else:
                return cv2.remap(self.last_frame, self.map1, self.map2,
                                 cv2.INTER_CUBIC)
        else:
            return None

    # def read_last_frame(self, distortion=False):
    #     if distortion:
    #         if self.last_distorted_frame is not None:
    #             return self.last_distorted_frame

    #     else:
    #         if self.last_frame is not None:
    #             return self.last_frame

    #     return None




