import numpy as np


def nms(boxes, scores, thresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = np.array(scores)

    order = np.argsort(scores)[-100:]
    keep_boxes = []
    while order.size > 0:
        i = order[-1]
        keep_boxes.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[:-1]] - inter)
        inds = np.where(ovr <= thresh)
        order = order[inds]

    return keep_boxes
