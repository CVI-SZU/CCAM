from __future__ import print_function

import cv2
import numpy as np

_GREEN = (18, 217, 15)
_RED = (15, 18, 217)


def vis_bbox(img, bbox, color=_GREEN, thick=1):
    '''Visualize a bounding box'''
    img = img.astype(np.uint8)
    (x0, y0, x1, y1) = bbox
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=thick)
    return img


def vis_one_image(img, boxes, color=_GREEN):
    for bbox in boxes:
        img = vis_bbox(img, (bbox[0], bbox[1], bbox[2], bbox[3]), color)
    return img
