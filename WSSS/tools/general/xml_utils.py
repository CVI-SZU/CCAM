# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import xml.etree.ElementTree as ET

def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    
    return bboxes, classes
