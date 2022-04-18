import os
import numpy as np
import PIL.Image
import torch
import torch.utils.data
import xml.etree.ElementTree as ET

def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lineSplit = line.strip().split(' ')
            imgPath, label = lineSplit[0], lineSplit[1]
            flag = lineSplit[2]
            imgList.append((imgPath, int(label), str(flag)))

    return imgList

def bboxes_reader(path):

    bboxes_list = {}
    bboxes_file = open(path + "/val.txt")
    for line in bboxes_file:
        line = line.split('\n')[0]
        line = line.split(' ')[0]
        labelIndex = line
        line = line.split("/")[-1]
        line = line.split(".")[0]+".xml"
        # bbox_path = path+"/val_boxes/" + line
        bbox_path = path+"/val_boxes/val/" + line
        tree = ET.ElementTree(file=bbox_path)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        bbox_line = []
        for Object in ObjectSet:
            BndBox = Object.find('bndbox')
            xmin = BndBox.find('xmin').text
            ymin = BndBox.find('ymin').text
            xmax = BndBox.find('xmax').text
            ymax = BndBox.find('ymax').text
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            bbox_line.append([xmin, ymin, xmax, ymax])
        bboxes_list[labelIndex] = bbox_line
    return bboxes_list

class ILSVRC2012(torch.utils.data.Dataset):
    def __init__(self, root, input_size, crop_size, train=True, transform=None):
        self._root = root
        self._train = train
        self._input_size = input_size
        self._crop_size = crop_size
        self._transform = transform
        self.loader = pil_loader

        if self._train:
            self.imgList = default_list_reader(self._root + '/train.txt')#[:1000]
        else:
            self.imgList = default_list_reader(self._root + '/val.txt')#[:1000]


        self.bboxes = bboxes_reader(self._root)

    def __getitem__(self, index):

        img_name, label, cls_name = self.imgList[index]

        image = self.loader(self._root+'/'+img_name)

        newBboxes = []
        if not self._train:
            bboxes = self.bboxes[img_name]
            for bbox_i in range(len(bboxes)):
                bbox = bboxes[bbox_i]
                # print(self._input_size, (self._input_size-self._crop_size)/2)
                bbox[0] = bbox[0] * (self._input_size / image.size[0]) - (self._input_size-self._crop_size)/2
                bbox[1] = bbox[1] * (self._input_size / image.size[1]) - (self._input_size-self._crop_size)/2
                bbox[2] = bbox[2] * (self._input_size / image.size[0]) - (self._input_size-self._crop_size)/2
                bbox[3] = bbox[3] * (self._input_size / image.size[1]) - (self._input_size-self._crop_size)/2
                bbox.insert(0, index)
                newBboxes.append(bbox)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)
        if self._train:
            return image, label, cls_name, img_name.split('/')[-1].split('.')[0]
        else:
            return image, label, newBboxes, cls_name, img_name.split('/')[-1].split('.')[0]  # only image name
            # return image, label, newBboxes, self._root+'/'+img_name

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.imgList)


def my_collate(batch):
    images = []
    labels = []
    bboxes = []
    cls_name = []
    img_name = []
    for sample in batch:
        images.append(sample[0])
        labels.append(torch.tensor(sample[1]))
        bboxes.append(torch.FloatTensor(sample[2]))
        cls_name.append(sample[3])
        img_name.append(sample[4])

    return torch.stack(images, 0), torch.stack(labels, 0), bboxes, cls_name, img_name

if __name__ == '__main__':

    import torchvision.transforms as transforms

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224,224)),
        #transforms.Resize(size=256),
        # transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # wrap to dataset
    test_data = ILSVRC2012(root='/data1/xjheng/dataset/ILSVRC2012', input_size=256, crop_size=224, train=False, transform=test_transforms)
    print(len(test_data))

    # train_data = ILSVRC2012(root='/data2/xjheng/dataset/ILSVRC2012', train=True, transform=test_transforms)
    # print(len(train_data))
    # [tensor([[63.6444, 35.5793, 167.7511, 177.8015],
    #          [-16.0000, 63.2652, 34.3467, 196.3852]])]

    # [tensor([[134.0000, -9.8560, 141.5976, 235.9040, 236.9249],
    #          [134.0000, 16.7680, 184.6487, 21.8880, 193.1051]])]

    # wrap to dataloader
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=my_collate)

    for i, (input, target, boxes, cls_name, img_name) in enumerate(test_loader):
        # print(i)
        print(cls_name, img_name)
        print(boxes)
        gtboxes = []
        # for j in range(len(boxes)):
        #         gtboxes.append(boxes[j][:, 1:])
        gtboxes = [boxes[k][:, 1:] for k in range(len(boxes))]
        print(gtboxes)
        break


