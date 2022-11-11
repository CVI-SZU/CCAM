import shutil
from torchvision.utils import make_grid
import torch
import cv2
import matplotlib

matplotlib.use('Agg')
import numpy as np
import json

import os, sys
import errno


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the acc@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)
    pred = (output > 0.5).long()
    acc = (pred.squeeze(1) == target).float().mean()
    return acc * 100


def find_bbox(scoremap, threshold=0.5):
    if isinstance(threshold, list):
        bboxes = []
        for i in range(len(threshold)):
            indices = np.where(scoremap > threshold[i])
            try:
                miny, minx = np.min(indices[0] * 4), np.min(indices[1] * 4)
                maxy, maxx = np.max(indices[0] * 4), np.max(indices[1] * 4)
            except:
                bboxes.append([0, 0, 0, 0])
            else:
                bboxes.append([minx, miny, maxx, maxy])
        return bboxes

    else:
        indices = np.where(scoremap > threshold)
        try:
            miny, minx = np.min(indices[0] * 4), np.min(indices[1] * 4)
            maxy, maxx = np.max(indices[0] * 4), np.max(indices[1] * 4)
        except:
            return [0, 0, 0, 0]
        else:
            return [minx, miny, maxx, maxy]


def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def IOUFunciton_ILSRVC(boxes_a, boxes_b):
    IOUList = np.zeros(len(boxes_b))
    for bbox_i in range(len(boxes_b)):  # #image
        box_a = boxes_a[bbox_i][0]
        box_a = torch.from_numpy(box_a).float()
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        imgBoxes_b = boxes_b[bbox_i]
        tempIOU = 0
        for bbox_j in range(imgBoxes_b.shape[0]):  # #boxes in one image
            box_b = imgBoxes_b[bbox_j].float()
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            intersect = (min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * (
                    min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
            abIOU = intersect / (area_a + area_b - intersect)
            if abIOU > tempIOU:
                tempIOU = abIOU
        IOUList[bbox_i] = tempIOU
    return torch.tensor(IOUList, dtype=torch.float)


import cmapy
def visualize_heatmap(config, experiments, images, attmaps, cls_name, image_name, phase='train', bboxes=None,
                      gt_bboxes=None):
    _, c, h, w = images.shape
    attmaps = attmaps.squeeze().to('cpu').detach().numpy()

    for i in range(images.shape[0]):

        # create folder
        if not os.path.exists('debug/images/{}/{}/colormaps/{}'.format(experiments, phase, cls_name[i])):
            os.mkdir('debug/images/{}/{}/colormaps/{}'.format(experiments, phase, cls_name[i]))

        attmap = attmaps[i]
        attmap = attmap / np.max(attmap)
        attmap = np.uint8(attmap * 255)
        # colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cv2.COLORMAP_JET)
        colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cmapy.cmap('seismic'))

        grid = make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0,
                         normalize=True, range=None)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]
        # print(image.shape, colormap.shape)
        cam = colormap + 0.5 * image
        cam = cam / np.max(cam)
        cam = np.uint8(cam * 255).copy()
        bbox_image = image.copy()

        if bboxes is not None:
            box = bboxes[i][0]

            cv2.rectangle(bbox_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)  # BGR

            if gt_bboxes is not None:
                if isinstance(gt_bboxes, list):
                    for j in range(gt_bboxes[i].shape[0]):
                        gtbox = gt_bboxes[i][j]
                        cv2.rectangle(bbox_image, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])),
                                      (255, 0, 0), 2)
                else:
                    gtbox = gt_bboxes[i]
                    cv2.rectangle(bbox_image, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])),
                                  (255, 0, 0),
                                  2)

        cv2.imwrite(f'debug/images/{experiments}/{phase}/colormaps/{cls_name[i]}/{image_name[i]}_raw.jpg', bbox_image)
        cv2.imwrite(f'debug/images/{experiments}/{phase}/colormaps/{cls_name[i]}/{image_name[i]}_heatmap.jpg', cam)


def save_bbox_as_json(config, experiments, cnt, rank, bboxes, cls_name, image_name, phase='train'):
    offset = (480 - 448) / 2
    pred_bbox = {}
    for i in range(len(bboxes)):
        box = bboxes[i][0]
        # save pseudo bboxes
        temp_bbox = [box[0] + offset, box[1] + offset, box[2] + offset, box[3] + offset]
        temp_bbox[2] = temp_bbox[2] - temp_bbox[0]
        temp_bbox[3] = temp_bbox[3] - temp_bbox[1]
        temp_save_box = [x / 480 for x in temp_bbox]
        if config.DATA == 'CUB_200_2011':
            key = f'{config.ROOT}/{phase}/{cls_name[i]}/{image_name[i]}.jpg'
            key = key.replace('//', '/')
            pred_bbox[key] = temp_save_box
        else:
            key = f'{config.ROOT}{phase}/{cls_name[i]}/{image_name[i]}.JPEG'
            key = key.replace('//', '/')
            pred_bbox[key] = temp_save_box

    with open(os.path.join(f'debug/images/{experiments}/{phase}/pseudo_boxes/{cnt}_{rank}_bbox.json'), 'w') as fp:
        json.dump(pred_bbox, fp)


# Plots a line-by-line description of a PyTorch model
def model_info(model):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float32:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def check_positive(am):
    am[am > 0.5] = 1
    am[am <= 0.5] = 0
    edge_mean = (am[0, 0, 0, :].mean() + am[0, 0, :, 0].mean() + am[0, 0, -1, :].mean() + am[0, 0, :, -1].mean()) / 4
    return edge_mean > 0.5


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list, factor,
                                  multi_contour_eval=False):
    """
    Copy from: https://github.com/clovaai/wsolevaluation
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """

    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)

            estimated_boxes.append([x0 * factor, y0 * factor, x1 * factor, y1 * factor])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list

def normalize_scoremap(alm):
    """
    Args:
        alm: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(alm).any():
        return np.zeros_like(alm)
    if alm.min() == alm.max():
        return np.zeros_like(alm)
    alm -= alm.min()
    alm /= alm.max()
    return alm


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

def creat_folder(config, args):
    # create folder for logging
    if not os.path.exists(config.DEBUG):
        os.mkdir(config.DEBUG)
        os.mkdir('{}/checkpoints'.format(config.DEBUG))
        os.mkdir('{}/images'.format(config.DEBUG))
    if not os.path.exists('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT))
    if not os.path.exists('{}/images/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/images/{}'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/train'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/test'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/train/colormaps'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/test/colormaps'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/train/pseudo_boxes'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/test/pseudo_boxes'.format(config.DEBUG, config.EXPERIMENT))
    if not os.path.exists('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT))
    shutil.copy(args.cfg, '{}/checkpoints/{}/{}'.format(config.DEBUG, config.EXPERIMENT, args.cfg.split('/')[-1]))
