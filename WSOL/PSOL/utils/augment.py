import math
import numbers
import random
import warnings

from PIL import Image
from torchvision.transforms import functional as F

from .func import *


# from torchvision.transforms import InterpolationMode

class RandomHorizontalFlipBBox(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            flipbox = copy.deepcopy(bbox)
            flipbox[0] = 1 - bbox[2]
            flipbox[2] = 1 - bbox[0]
            return F.hflip(img), flipbox

        return img, bbox


class RandomResizedBBoxCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, bbox, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """

        area = img.size[0] * img.size[1]

        for attempt in range(30):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:

                i = random.randint(0, img.size[1] - h)  # i is y actually
                j = random.randint(0, img.size[0] - w)  # j is x

                # compute intersection between crop image and bbox
                intersec = compute_intersec(i, j, h, w, bbox)

                if intersec[2] - intersec[0] > 0 and intersec[3] - intersec[1] > 0:
                    intersec = normalize_intersec(i, j, h, w, intersec)
                    return i, j, h, w, intersec

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]

        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2

        intersec = compute_intersec(i, j, h, w, bbox)
        intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, h, w, intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, crop_bbox = self.get_params(img, bbox, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), crop_bbox


class RandomBBoxCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, bbox, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        intersec = compute_intersec(i, j, h, w, bbox)
        intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w, crop_bbox = self.get_params(img, bbox, self.size)

        return F.crop(img, i, j, h, w), crop_bbox

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ResizedBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        # resize to 256
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
        else:
            ow, oh = size[::-1]
            w, h = img.size

        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h
        intersec[0] = bbox[0] * ratew
        intersec[2] = bbox[2] * ratew
        intersec[1] = bbox[1] * rateh
        intersec[3] = bbox[3] * rateh

        # intersec = normalize_intersec(i, j, h, w, intersec)
        return (oh, ow), intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox


class CenterBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        # center crop
        if isinstance(size, numbers.Number):
            output_size = (int(size), int(size))

        w, h = img.size
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        intersec = compute_intersec(i, j, th, tw, bbox)
        intersec = normalize_intersec(i, j, th, tw, intersec)

        # intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, th, tw, crop_bbox = self.get_params(img, bbox, self.size)
        return F.center_crop(img, self.size), crop_bbox
