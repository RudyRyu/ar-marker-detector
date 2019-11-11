
import cv2
import numpy as np
import threading
from threading import Lock

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()
    K = np.array([[9784.521+232.291, 0., 959.784],
              [0, 9432.565+225.503, 758.442],
              [0, 0, 1]])

    # Distortion Coefficients(kc) - 1st, 2nd
    d = np.array([-16.271484, 279.268252, -0.134332, 0.149862, 0]) # just use first two terms

    def __init__(self, rtsp_link, img_size):
        cam_mat, roi = cv2.getOptimalNewCameraMatrix(self.K, self.d, img_size, 0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.d, None, cam_mat, img_size, 5)
        capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(capture,), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()

    def read(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return cv2.remap(self.last_frame.copy(), self.map1, self.map2, cv2.INTER_CUBIC)
        else:
            return None