# modified by Sierkinhane <sierkinhane@163.com>
import json
import math
import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image
from utils.augment import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_bbox_dict(root):
    print('loading from ground truth bbox')
    name_idx_dict = {}
    with open(os.path.join(root, 'images.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, name = fileline[0], fileline[1]
            name_idx_dict[name] = idx

    idx_bbox_dict = {}
    with open(os.path.join(root, 'bounding_boxes.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, bbox = fileline[0], list(map(float, fileline[1:]))
            idx_bbox_dict[idx] = bbox

    name_bbox_dict = {}
    for name in name_idx_dict.keys():
        name_bbox_dict[name] = idx_bbox_dict[name_idx_dict[name]]

    return name_bbox_dict


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def load_train_bbox(bbox_dir):
    final_dict = {}
    file_list = os.listdir(bbox_dir)
    for i, now_name in enumerate(file_list):
        if i % 1000 == 0:
            print(f'load [{i}/{len(file_list)}] json!')
        now_json_file = os.path.join(bbox_dir, now_name)
        with open(now_json_file, 'r') as fp:
            name_bbox_dict = json.load(fp)
        final_dict.update(name_bbox_dict)

    return final_dict


def load_val_bbox(all_imgs, gt_location):
    import scipy.io as sio
    gt_label = sio.loadmat(os.path.join(gt_location, 'cache_groundtruth.mat'))
    locs = [(x[0].split('/')[-1], x[0], x[1]) for x in all_imgs]
    locs.sort()
    final_bbox_dict = {}
    for i in range(len(locs)):
        # gt_label['rec'][:,1][0][0][0], if multilabel then get length, for final eval
        final_bbox_dict[locs[i][1]] = gt_label['rec'][:, i][0][0][0][0][1][0]
    return final_bbox_dict


class CUBDataset(data.Dataset):

    def __init__(self, root, pseudo_bboxes_path, input_size=256, crop_size=224, train=True, transform=None,
                 target_transform=None, loader=default_loader):
        from torchvision.datasets import ImageFolder
        self.train = train
        self.input_size = input_size
        self.crop_size = crop_size
        self.pseudo_bboxes_path = pseudo_bboxes_path
        if self.train:
            self.img_dataset = ImageFolder(os.path.join(root, 'train'))
        else:
            self.img_dataset = ImageFolder(os.path.join(root, 'test'))
        if len(self.img_dataset) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.label_class_dict = {}
        self.train = train

        for k, v in self.img_dataset.class_to_idx.items():
            self.label_class_dict[v] = k
        if self.train:
            # load train bbox
            self.bbox_dict = load_train_bbox(self.pseudo_bboxes_path)

        self.img_dataset = self.img_dataset.imgs

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.img_dataset[index]
        img = self.loader(path)
        if self.train:
            bbox = self.bbox_dict[path]
        else:
            bbox = self.bbox_dict[path]
        w, h = img.size

        bbox = np.array(bbox, dtype='float32')

        # convert from x, y, w, h to x1,y1,x2,y2
        if self.train:
            bbox[0] = bbox[0]
            bbox[2] = bbox[0] + bbox[2]
            bbox[1] = bbox[1]
            bbox[3] = bbox[1] + bbox[3]
            bbox[0] = math.ceil(bbox[0] * w)
            bbox[2] = math.ceil(bbox[2] * w)
            bbox[1] = math.ceil(bbox[1] * h)
            bbox[3] = math.ceil(bbox[3] * h)

            # visualize bbox
            # import cv2
            # imgc = cv2.rectangle(np.uint8(img), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # import os
            # if not os.path.exists('./test_visual_bboxes'):
            #     os.mkdir('./test_visual_bboxes')
            # cv2.imwrite(f'./test_visual_bboxes/{index}.png', imgc)

            img_i, bbox_i = RandomResizedBBoxCrop((self.crop_size))(img, bbox)
            img, bbox = RandomHorizontalFlipBBox()(img_i, bbox_i)
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]

        else:
            img_i, bbox_i = ResizedBBoxCrop((self.input_size, self.input_size))(img, bbox)
            img, bbox = CenterBBoxCrop((self.crop_size))(img_i, bbox_i)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, bbox
        else:
            return img, target

    def __len__(self):
        return len(self.img_dataset)


if __name__ == '__main__':
    pass
