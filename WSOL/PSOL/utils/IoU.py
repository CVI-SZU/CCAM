import xml.etree.ElementTree as ET

import torch


def get_gt_boxes(xmlfile):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        gt_boxes.append((x1, y1, x2, y2))
    return gt_boxes


def get_cls_gt_boxes(xmlfile, cls):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        cls_name = obj.find('name').text
        # print(cls_name, cls)
        if cls_name != cls:
            continue
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        gt_boxes.append((x1, y1, x2, y2))
    if len(gt_boxes) == 0:
        pass
        # print('%s bbox = 0'%cls)

    return gt_boxes


def get_cls_and_gt_boxes(xmlfile, cls, class_to_idx):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        cls_name = obj.find('name').text
        # print(cls_name, cls)
        if cls_name != cls:
            continue
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        gt_boxes.append((class_to_idx[cls_name], [x1, y1, x2 - x1, y2 - y1]))
    if len(gt_boxes) == 0:
        pass
        # print('%s bbox = 0'%cls)

    return gt_boxes


def convert_boxes(boxes):
    ''' convert the bbox to the format (x1, y1, x2, y2) where x1,y1<x2,y2'''
    converted_boxes = []
    for bbox in boxes:
        (x1, y1, x2, y2) = bbox
        converted_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return converted_boxes


def IoU(a, b):
    # print(a, b)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    def compute_area(box):
        dx = max(0, box[2] - box[0])
        dy = max(0, box[3] - box[1])
        dx = float(dx)
        dy = float(dy)
        return dx * dy

    # print(x1, y1, x2, y2)
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    # inter = w*h
    # aarea = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    # barea = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    inter = compute_area([x1, y1, x2, y2])
    aarea = compute_area(a)
    barea = compute_area(b)

    # assert aarea+barea-inter>0
    if aarea + barea - inter <= 0:
        print(a)
        print(b)
    o = inter / (aarea + barea - inter)
    # if w<=0 or h<=0:
    #    o = 0
    return o


def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp


def xywh_to_x1y1x2y2(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes


def x1y1x2y2_to_xywh(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes


def compute_IoU(pred_box, gt_box):
    boxes1 = to_2d_tensor(pred_box)
    # boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes1[:, 2] = torch.clamp(boxes1[:, 0] + boxes1[:, 2], 0, 1)
    boxes1[:, 3] = torch.clamp(boxes1[:, 1] + boxes1[:, 3], 0, 1)

    boxes2 = to_2d_tensor(gt_box)
    boxes2[:, 2] = torch.clamp(boxes2[:, 0] + boxes2[:, 2], 0, 1)
    boxes2[:, 3] = torch.clamp(boxes2[:, 1] + boxes2[:, 3], 0, 1)
    # boxes2 = xywh_to_x1y1x2y2(boxes2)

    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert ((a1 + a2 - ia < 0).sum() == 0)
    return ia / (a1 + a2 - ia)
