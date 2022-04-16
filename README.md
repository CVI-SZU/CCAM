# CCAM

Code repository for our
paper "[CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation](https://arxiv.org/pdf/2203.13505.pdf)"
in CVPR 2022.

![](images/CCAM_Network.png)

The repository includes full training, evaluation, and visualization codes
on [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200.html), [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/), and [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) datasets.

## Dependencies

* Python 3
* PyTorch 1.7.1
* OpenCV-Python
* Numpy
* Scipy
* MatplotLib
* Yaml
* Easydict

## Dataset

### CUB-200-2011

You will need to download the images (JPEG format) in CUB-200-2011 dataset
at [here](http://www.vision.caltech.edu/visipedia/CUB-200.html). Make sure your ```data/CUB_200_2011``` folder is structured as
follows:

```
├── CUB_200_2011/
|   ├── images
|   ├── images.txt
|   ├── bounding_boxes.txt
|   ...
|   └── train_test_split.txt
```

You will need to download the images (JPEG format) in ILSVRC2012 dataset at [here](https://image-net.org/challenges/LSVRC/2012/).
Make sure your ```data/ILSVRC2012``` folder is structured as follows:

### ILSVRC2012

```
├── ILSVRC2012/ 
|   ├── train
|   ├── val
|   ├── val_boxes
|   |   ├——val
|   |   |   ├—— ILSVRC2012_val_00050000.xml
|   |   |   ├—— ...
|   ├── train.txt
|   └── val.txt
```

### PASCAL VOC2012

You will need to download the images (JPEG format) in PASCAL VOC2012 dataset at [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
Make sure your ```data/VOC2012``` folder is structured as follows:

```
├── VOC2012/
|   ├── Annotations
|   ├── ImageSets
|   ├── SegmentationClass
|   ├── SegmentationClassAug
|   └── SegmentationObject
```

## For WSOL task

please refer to the director of './WSOL'

```
cd WSOL
```

## For WSSS task

please refer to the director of './WSSS'

```
cd WSSS
```

## CUSTOM DATASET

```
cd CUSTOM
```

### Comparison with CAM

![](images/CCAM_Heatmap.png)

### Extracted Background Cues

![](images/CCAM_Background.png)

## Reference

If you are using our code, please consider citing our paper.

```
@InProceedings{CCAM,
    author    = {Xie, Jinheng and Xiang, Jianfeng and Chen, Junliang and Hou, Xianxu and Zhao, Xiaodong and Shen, Linlin},
    title     = {CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation},
    booktitle = {CVPR},
    year      = {2022},
}
```
