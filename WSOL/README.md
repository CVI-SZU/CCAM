### For WSOL task

Download the pretrained parameters (e.g., moco and detco) from [here](https://drive.google.com/drive/folders/1erzARKq9g02-3pUGhY6-hyGzD-hoty5b?usp=sharing) and put them in the current directory.

```
├── WSOL/
|   ├── config
|   ├—— ...
|   ├—— moco_r50_v2-e3b0c442.pth
|   └── detco_200ep.pth
```

#### CUB-200-2011
Train CCAM on **CUB-200-2011** dataset (supervised parameters) 

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_CUB.py --experiment CCAM_CUB_IP --lr 0.0001 --batch_size 16 --pretrained supervised --alpha 0.05
```

Train CCAM on CUB-200-2011 dataset (unsupervised parameters) 

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_CUB.py --experiment CCAM_CUB_MOCO --lr 0.0001 --batch_size 16 --pretrained mocov2 --alpha 0.75
```

or

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_CUB.py --experiment CCAM_CUB_DETCO --lr 0.0001 --batch_size 16 --pretrained detco --alpha 0.75
```

The code will create experiment folders for model checkpoints (./debug/checkpoint), log files (./log) and visualization (./debug/images/).

```
├── debug/
|   ├── checkpoints
|   ├—— images
|   |   ├—— CCAM_CUB_MOCO
|   |   |   ├—— train
|   |   |   |   ├—— colormaps
|   |   |   |   ├—— pseudo_boxes
|   |   |   ├—— test
|   |   ├—— ...
```

#### ILSVRC2012
Train CCAM on ILSVRC dataset (supervised parameters) 

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_ILSVRC.py --experiment CCAM_ILSVRC_IP --lr 0.0001 --batch_size 256 --pretrained supervised --alpha 0.05 --port 2345
```

Train CCAM on ILSVRC dataset (unsupervised parameters) 

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_ILSVRC.py --experiment CCAM_ILSVRC_MOCO --lr 0.0001 --batch_size 256 --pretrained mocov2 --alpha 0.05 --port 2345
```

or

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_ILSVRC.py --experiment CCAM_ILSVRC_DETCO --lr 0.0001 --batch_size 256 --pretrained detco --alpha 0.05 --port 2345
```

The code will create experiment folders for model checkpoints (./debug/checkpoint), log files (./log) and visualization (./debug/images/).

```
├── debug/
|   ├── checkpoints
|   ├—— images
|   |   ├—— CCAM_ILSVRC_MOCO
|   |   |   ├—— train
|   |   |   |   ├—— colormaps
|   |   |   |   ├—— pseudo_boxes
|   |   |   ├—— test
|   |   ├—— ...
```

**Note that** we use a tesla A100 to train the model and please specify more GPUs to prevent the problem of OOM when training on ILSVRC2012 dataset.

We also provide the extracted class-agnositc bounding boxes from [here](https://drive.google.com/drive/folders/1erzARKq9g02-3pUGhY6-hyGzD-hoty5b).

**To train a regressor using the extracted pseudo bboxes, please refer to directory `./PSOL`**

### Reference

If you are using our code, please consider citing our paper.

```
@InProceedings{Xie_2022_CVPR,
    author    = {Xie, Jinheng and Xiang, Jianfeng and Chen, Junliang and Hou, Xianxu and Zhao, Xiaodong and Shen, Linlin},
    title     = {{C2AM}: Contrastive Learning of Class-Agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {989-998}
}
@article{xie2022contrastive,
  title={Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation},
  author={Xie, Jinheng and Xiang, Jianfeng and Chen, Junliang and Hou, Xianxu and Zhao, Xiaodong and Shen, Linlin},
  journal={arXiv preprint arXiv:2203.13505},
  year={2022}
}
```

