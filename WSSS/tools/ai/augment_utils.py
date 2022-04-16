import cv2
import random
import numpy as np

from PIL import Image

def convert_OpenCV_to_PIL(image):
    return Image.fromarray(image[..., ::-1])

def convert_PIL_to_OpenCV(image):
    return np.asarray(image)[..., ::-1]

class RandomResize:
    def __init__(self, min_image_size, max_image_size):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size

        self.modes = [Image.BICUBIC, Image.NEAREST]
    
    def __call__(self, image, mode=Image.BICUBIC):
        rand_image_size = random.randint(self.min_image_size, self.max_image_size)
        
        w, h = image.size
        if w < h:
            scale = rand_image_size / h
        else:
            scale = rand_image_size / w

        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            return image

        return image.resize(size, mode)

class RandomResize_For_Segmentation:
    def __init__(self, min_image_size, max_image_size):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        
        self.modes = [Image.BICUBIC, Image.NEAREST]
    
    def __call__(self, data):
        image, mask = data['image'], data['mask']

        rand_image_size = random.randint(self.min_image_size, self.max_image_size)
        
        w, h = image.size
        if w < h:
            scale = rand_image_size / h
        else:
            scale = rand_image_size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            pass
        else:
            data['image'] = image.resize(size, Image.BICUBIC)
            data['mask'] = mask.resize(size, Image.NEAREST)

        return data

class RandomHorizontalFlip:
    def __init__(self):
        pass

    def __call__(self, image):
        if bool(random.getrandbits(1)):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

class RandomHorizontalFlip_For_Segmentation:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if bool(random.getrandbits(1)):
            data['image'] = image.transpose(Image.FLIP_LEFT_RIGHT)
            data['mask'] = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return data

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.asarray(image)
        norm_image = np.empty_like(image, np.float32)

        norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        return norm_image

class Normalize_For_Segmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)

        norm_image = np.empty_like(image, np.float32)

        norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        data['image'] = norm_image
        data['mask'] = mask

        return data

class Top_Left_Crop:
    def __init__(self, crop_size, channels=3):
        self.bg_value = 0
        self.crop_size = crop_size
        self.crop_shape = (self.crop_size, self.crop_size, channels)

    def __call__(self, image):
        h, w, c = image.shape

        ch = min(self.crop_size, h)
        cw = min(self.crop_size, w)

        cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
        cropped_image[:ch, :cw] = image[:ch, :cw]
        
        return cropped_image

class Top_Left_Crop_For_Segmentation:
    def __init__(self, crop_size, channels=3):
        self.bg_value = 0
        self.crop_size = crop_size
        self.crop_shape = (self.crop_size, self.crop_size, channels)
        self.crop_shape_for_mask = (self.crop_size, self.crop_size)

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        h, w, c = image.shape

        ch = min(self.crop_size, h)
        cw = min(self.crop_size, w)

        cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
        cropped_image[:ch, :cw] = image[:ch, :cw]
        
        cropped_mask = np.ones(self.crop_shape_for_mask, mask.dtype) * 255
        cropped_mask[:ch, :cw] = mask[:ch, :cw]

        data['image'] = cropped_image
        data['mask'] = cropped_mask

        return data


class RandomCrop:
    def __init__(self, crop_size, channels=3, with_bbox=False):
        self.bg_value = 0
        self.with_bbox = with_bbox
        self.crop_size = crop_size
        self.crop_shape = (self.crop_size, self.crop_size, channels)

    def get_random_crop_box(self, image):
        h, w, c = image.shape

        ch = min(self.crop_size, h)
        cw = min(self.crop_size, w)

        w_space = w - self.crop_size
        h_space = h - self.crop_size

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space + 1)
        else:
            cont_left = random.randrange(-w_space + 1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space + 1)
        else:
            cont_top = random.randrange(-h_space + 1)
            img_top = 0

        dst_bbox = {
            'xmin' : cont_left, 'ymin' : cont_top,
            'xmax' : cont_left+cw, 'ymax' : cont_top+ch
        }
        src_bbox = {
            'xmin' : img_left, 'ymin' : img_top,
            'xmax' : img_left+cw, 'ymax' : img_top+ch
        }

        return dst_bbox, src_bbox
    
    def __call__(self, image, bbox_dic=None):
        if bbox_dic is None:
            dst_bbox, src_bbox = self.get_random_crop_box(image)
        else:
            dst_bbox, src_bbox = bbox_dic['dst_bbox'], bbox_dic['src_bbox']
        
        cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
        cropped_image[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
            image[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]

        if self.with_bbox:
            return cropped_image, {'dst_bbox':dst_bbox, 'src_bbox':src_bbox}
        else:
            return cropped_image

class RandomCrop_For_Segmentation(RandomCrop):
    def __init__(self, crop_size):
        super().__init__(crop_size)

        self.crop_shape_for_mask = (self.crop_size, self.crop_size)

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        dst_bbox, src_bbox = self.get_random_crop_box(image)
        
        cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
        cropped_image[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
            image[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]

        cropped_mask = np.ones(self.crop_shape_for_mask, mask.dtype) * 255
        cropped_mask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
            mask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        
        data['image'] = cropped_image
        data['mask'] = cropped_mask
        
        return data

class Transpose:
    def __init__(self):
        pass
        
    def __call__(self, image):
        return image.transpose((2, 0, 1))

class Transpose_For_Segmentation:
    def __init__(self):
        pass
        
    def __call__(self, data):
        # h, w, c -> c, h, w
        data['image'] = data['image'].transpose((2, 0, 1))
        return data

class Resize_For_Mask:
    def __init__(self, size):
        self.size = (size, size)
    
    def __call__(self, data):
        mask = Image.fromarray(data['mask'].astype(np.uint8))
        mask = mask.resize(self.size, Image.NEAREST)
        data['mask'] = np.asarray(mask, dtype=np.uint64)
        return data
