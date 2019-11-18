from __future__ import print_function

import argparse
import json
import socket
import time
from collections import defaultdict
from pprint import pprint

import cv2

from ar_marker_detector.detect import detect_markers
from common.cam_reader import Camera
from common.utils import check_time
from common.utils import log
from common.utils import arg_parser


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
def get_map_coords(map_coords, map_markers):

    def get_map_coord(i, j, k):

        """
        Args:
            i: map marker position 1
            j: map marker position 2
            k: idx of map_coords dictionary

        Return:
            (x1, x2, y1 ,y2)
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
def get_obj_absolute_coords(obj_markers, x1, x2, y1, y2):

    def get_obj_absolute_coord(obj_marker):
        try:
            obj_abs_x = int(x1[0]+(x2[0]-x1[0]) \
                             * ((obj_marker.center[0]-x1[1])/(x2[1]-x1[1])))
            obj_abs_y = int(y1[0]+(y2[0]-y1[0]) \
                             * ((obj_marker.center[1]-y1[1])/(y2[1]-y1[1])))
        except ZeroDivisionError as e:
            log.info(e)
            obj_abs_x = 0
            obj_abs_y = 0

        return (obj_abs_x, obj_abs_y)

    obj_abs_coords = []
    for obj_marker in obj_markers:
        obj_abs_coords.append((obj_marker, get_obj_absolute_coord(obj_marker)))

    return obj_abs_coords


@check_time
def set_final_obj_abs_coords_per_cam(final_obj_abs_coords, obj_abs_coords,
                                                           img):
    for oac in obj_abs_coords:
        final_obj_abs_coords[oac[0].id].append(oac[1])
        text = f'  {oac[1]}'
        cv2.putText(img, text, oac[0].center,
                    1, 4.0, (0, 255, 0), thickness=5)


@check_time
def show_img(img_name, img, view_size, r_scale=1):
    cv2.imshow(img_name, cv2.resize(img, view_size, fx=r_scale, fy=r_scale))


@check_time
def set_final_obj_abs_coords(final_obj_abs_coords):
    for idx in final_obj_abs_coords.keys():
        overlap_num = len(final_obj_abs_coords[idx])
        if overlap_num == 1:
            final_obj_abs_coords[idx] = final_obj_abs_coords[idx][0]

        elif overlap_num > 1:
            x = 0
            y = 0
            for f, abs_coord in enumerate(final_obj_abs_coords[idx]):
                # log.debug(f'{f}: {abs_coord}')
                x += abs_coord[0]
                y += abs_coord[1]

            final_obj_abs_coords[idx] = \
                        (int(x/overlap_num), int(y/overlap_num))


@check_time
def send_data(final_obj_abs_coords, car_ids, obs_ids,
              udp_sock, server_address):

    json_data = {'car': {}, 'obs': {}}
    if final_obj_abs_coords:
        for idx in final_obj_abs_coords.keys():
            if idx in car_ids:
                json_data['car'][idx]=final_obj_abs_coords[idx]
            elif idx in obs_ids:
                json_data['obs'][idx]=final_obj_abs_coords[idx]

    json_string = json.dumps(json_data)
    log.info(json_string)

    # json_bytes = str.encode(json_string)
    # udp_sock.sendto(json_bytes, server_address)


def run_detection(conf):

    map_infos = {}
    for key in conf['map_infos']:
        map_infos[int(key)] = conf['map_infos'][key]

    car_ids = conf['car_ids']
    obs_ids = conf['obs_ids']

    img_size = tuple(conf['img_size'])

    cams = []
    for cam_info in conf['cam_infos']:
        cam = Camera(
                name=cam_info,
                rtsp_link=conf['cam_infos'][cam_info]['rtsp_link'],
                map_positions=conf['cam_infos'][cam_info]['map_positions'],
                img_size=img_size,
                calibration=conf['cam_infos'][cam_info]['calibration'],
                flip=conf['cam_infos'][cam_info]['flip'],
               )

        cams.append(cam)


    view_size = tuple(conf['view_size'])
    r_scale = conf['resize_scale']

    udp_sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    server_address = tuple(conf['server_address'])

    while True:
        final_obj_abs_coords = defaultdict(lambda: [])

        s = time.time()
        for cam in cams:
            log.debug('')
            img = cam.read_last_frame()
            if img is None:
                continue

            detected_markers, contours = detect_markers(img, scale=1)
            # img = draw_contours(img, contours)
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
                    show_img(cam.name, img, view_size)
                    continue
                else:
                    map_coords = cam.last_map_pix
            else:
                cam.last_map_pix = map_coords

            obj_abs_coords = get_obj_absolute_coords(
                                    [*car_markers, *obs_markers], *map_coords)
            set_final_obj_abs_coords_per_cam(final_obj_abs_coords,
                                             obj_abs_coords, img)

            show_img(cam.name, img, view_size)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        set_final_obj_abs_coords(final_obj_abs_coords)
        send_data(final_obj_abs_coords, car_ids, obs_ids,
                  udp_sock, server_address)

        log.debug(f'Final time {time.time()-s:.3f} sec')

if __name__ == '__main__':

    args = arg_parser.parse_args()
    config_path = args.conf

    with open(config_path) as config_buffer:
        conf = json.loads(config_buffer.read())

    run_detection(conf)
