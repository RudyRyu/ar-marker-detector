import threading
import time
from threading import Lock

import cv2
import numpy as np


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
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 2010)

        thread = threading.Thread(target=self.rtsp_cam_buffer,
                                  args=(self.capture,),
                                  name="rtsp_read_thread")
        thread.daemon = True
        # thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()
                # time.sleep(0.01)


    def read_each_img(self):
        ret, img = self.capture.read()
        if ret:
            img = cv2.remap(img.copy(), self.map1, self.map2, cv2.INTER_CUBIC)

        return ret, img


    def read(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return cv2.remap(self.last_frame.copy(), self.map1,
                             self.map2, cv2.INTER_CUBIC)
        else:
            return None
