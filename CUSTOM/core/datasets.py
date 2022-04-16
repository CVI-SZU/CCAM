import os
import torch
from PIL import Image
from tools.ai.augment_utils import *


class CUSTOM_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_name_list = []
        for i in os.listdir(self.data_dir):
            if i.endswith('.jpg') or i.endswith('.png'):
                self.image_name_list.append(i)

    def __len__(self):
        return len(self.image_name_list)

    def get_image(self, image_name):
        image = Image.open(self.data_dir + image_name).convert('RGB')
        return image

    def __getitem__(self, index):
        image_name = self.image_name_list[index]

        image = self.get_image(image_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name

class CUSTOM_Dataset_For_Making_CAM(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_name_list = []
        for i in os.listdir(self.data_dir):
            if i.endswith('.jpg') or i.endswith('.png'):
                self.image_name_list.append(i)

    def __len__(self):
        return len(self.image_name_list)

    def get_image(self, image_name):
        image = Image.open(self.data_dir + image_name).convert('RGB')
        return image

    def __getitem__(self, index):
        image_name = self.image_name_list[index]

        image = self.get_image(image_name)

        return image, image_name
