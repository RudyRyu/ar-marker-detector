from __future__ import print_function

import argparse
from collections import defaultdict

import cv2

from ar_marker_detector.detect import detect_markers
from common.cam_reader import Camera
from common.utils import check_time
from common.utils import log


# parser = argparse.ArgumentParser()
# parser.add_argument('-cam')
# parser.add_argument('-frame_size', type=int, nargs='+')

@check_time
def classify_markers(detected_markers, marker_positions, car_ids, obs_ids):

    map_markers_temp = []
    car_markers = []
    obs_markers = []
    for dm in detected_markers:
        if dm.id in marker_positions:
            map_markers_temp.append(dm)
        elif dm.id in car_ids:
            car_markers.append(dm)
        elif dm.id in obs_ids:
            obs_markers.append(dm)

    map_markers = []
    for mp in marker_positions:
        if mp is None:
            map_markers.append(None)
            continue

        for mmt in map_markers_temp:
            if mmt.id == mp:
                map_markers.append(mmt)
                break
        else:
            map_markers.append(None)

    return map_markers, car_markers, obs_markers


@check_time
def draw_contours(img, rects):
    img_copy = img.copy()
    for rect in rects:
        cv2.drawContours(img_copy, [rect], -1, (0, 0, 255), 5)

    return img_copy

@check_time
def draw_marker_boxes(img, detected_markers):
    img_copy = img.copy()
    for dm in detected_markers:
        dm.highlight_marker(img_copy)

    return img_copy

@check_time
def get_map_coords(map_coords, map_markers):

    def get_map_coord(i, j, k):

        """
        Args:
            i: map marker position 1
            j: map marker position 2
            k: idx of map_coords dictionary

        Return:
            (map position abs, map position pixel)
        """

        if (map_markers[i] is None) and (map_markers[j] is None):
            val = None

        elif map_markers[i] is not None:
            val = (map_coords[map_markers[i].id][k], map_markers[i].center[k])

        elif map_markers[j] is not None:
            val = (map_coords[map_markers[j].id][k], map_markers[j].center[k])

        else:
            map_coord = int((map_coords[map_markers[i].id][k] \
                              + map_coords[map_markers[j].id][k]) / 2)
            marker_center = int((map_markers[i].center[k] \
                                  + map_markers[j].center[k]) / 2)
            val = (map_coord, marker_center)

        return val

    x1 = get_map_coord(0,2,0)
    x2 = get_map_coord(1,3,0)
    y1 = get_map_coord(0,1,1)
    y2 = get_map_coord(2,3,1)

    if None in (x1,x2,y1,y2):
        return None
    else:
        return (x1,x2,y1,y2)

@check_time
def get_car_absolute_coords(car_markers, x1, x2, y1, y2):

    def get_car_absolute_coord(car_marker):

        car_abs_x = int(x1[0]+(x2[0]-x1[0]) \
                         * ((car_marker.center[0]-x1[1])/(x2[1]-x1[1])))
        car_abs_y = int(y1[0]+(y2[0]-y1[0]) \
                         * ((car_marker.center[1]-y1[1])/(y2[1]-y1[1])))

        return (car_abs_x, car_abs_y)

    car_abs_coords = []
    for car_marker in car_markers:
        car_abs_coords.append((car_marker, get_car_absolute_coord(car_marker)))

    return car_abs_coords

@check_time
def set_final_car_abs_coords_per_cam(final_car_abs_coords, car_abs_coords,
                                                           img):
    for cac in car_abs_coords:
        final_car_abs_coords[cac[0].id].append(cac[1])
        text = f'  {cac[1]}'
        cv2.putText(img, text, cac[0].center,
                    1, 4.0, (255, 0, 0), thickness=5)

@check_time
def show_img(img_name, img, r_scale):
    cv2.imshow(img_name, cv2.resize(img, None, fx=r_scale, fy=r_scale))
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

@check_time
def set_final_car_abs_coords(final_car_abs_coords):
    for idx in final_car_abs_coords.keys():
        overlap_num = len(final_car_abs_coords[idx])
        if overlap_num == 1:
            final_car_abs_coords[idx] = final_car_abs_coords[idx][0]

        elif overlap_num > 1:
            x = 0
            y = 0
            for f, abs_coord in enumerate(final_car_abs_coords[idx]):
                log.debug(f'{f}: {abs_coord}')
                x += abs_coord[0]
                y += abs_coord[1]

            final_car_abs_coords[idx] = \
                        (int(x/overlap_num), int(y/overlap_num))

@check_time
def something_to_do(final_car_abs_coords):
    if final_car_abs_coords:
        for idx in final_car_abs_coords.keys():
            log.info(f'final: {idx} {final_car_abs_coords[idx]}')


def run_detection(conf):

    # args = parser.parse_args()
    # capture = Camera(args.cam, tuple(args.frame_size))

    map_infos = {2:(0,0),
                 4:(57,0),
                 1:(0,56),
                 3:(57,56)}

    car_ids = [101, 102]
    obs_ids = []

    cam1 = Camera(name='cam1',
                  rtsp_link='demo.mkv',
                  map_positions=(2,4,1,3),
                  last_map_pix=None,
                  img_size=(1920, 1080))

    # cam2 = Camera(~)

    cams = [cam1]
    view_size = (1280, 720)

    while True:
        final_car_abs_coords = defaultdict(lambda: [])
        for cam in cams:
            img = cam.read()
            if img is None:
                continue

            detected_markers, contours = detect_markers(img)
            img = draw_contours(img, contours)
            img = draw_marker_boxes(img, detected_markers)
            map_markers, car_markers, obs_markers = \
                                    classify_markers(detected_markers,
                                                     cam.map_positions,
                                                     car_ids,
                                                     obs_ids)

            map_coords = get_map_coords(map_infos, map_markers)
            if map_coords is None:
                if cam.last_map_pix is None:
                    log.info('Map markers must be detected')
                    show_img(cam.name, img, r_scale)
                    continue
                else:
                    map_coords = cam.last_map_pix
            else:
                cam.last_map_pix = map_coords

            car_abs_coords = get_car_absolute_coords(car_markers, *map_coords)
            set_final_car_abs_coords_per_cam(final_car_abs_coords,
                                             car_abs_coords, img)

            resized = cv2.resize(img, view_size)
            cv2.imshow('Result', resized)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    run_detection(None)

