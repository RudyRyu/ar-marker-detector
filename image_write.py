import cv2
import numpy as np

cap1 = cv2.VideoCapture("rtsp://admin:ijoonn13407@192.168.3.33/h264")
cap2 = cv2.VideoCapture("rtsp://admin:ijoonn13407@192.168.3.37/h264")

print(cap1.get(cv2.CAP_PROP_FPS))
print(cap1.get(3))
print(cap1.get(4))

print(cap2.get(cv2.CAP_PROP_FPS))
print(cap2.get(3))
print(cap2.get(4))

frame_num = 1800
img_list1 = np.empty((frame_num, 1080, 1920, 3), dtype=np.uint8)
img_list2 = np.empty((frame_num, 1080, 1920, 3), dtype=np.uint8)

print('run_start')
for i in range(frame_num):
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey()

    if ret1:
        img_list1[i] = img1[:]
        # cv2.imshow('img1', cv2.resize(img1, None, fx=0.5, fy=0.5))
    if ret2:
        img_list2[i] = img2[:]
        # cv2.imshow('img2', cv2.resize(img2, None, fx=0.5, fy=0.5))

out1 = cv2.VideoWriter('output1.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30,
                      (int(cap1.get(3)), int(cap1.get(4))))

out2 = cv2.VideoWriter('output2.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30,
                      (int(cap2.get(3)), int(cap2.get(4))))

print('cam1 save start')
for i in range(frame_num):
    out1.write(img_list1[i])

print('cam2 save start')
for i in range(frame_num):
    out2.write(img_list2[i])

cap1.release()
cap2.release()
out1.release()
out2.release()

cv2.destroyAllWindows()
