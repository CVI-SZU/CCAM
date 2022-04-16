### For WSOL task

Download the pretrained parameters (e.g., moco and detco) at [here](https://drive.google.com/file/d/1W9f8Jy0m-SOurvU1sFLvp4--xIKCvpzB/view) and put them in the current directory.

```
├── WSOL/
|   ├── config
|   ├—— ...
|   ├—— moco_r50_v2-e3b0c442.pth
|   └── detco_200ep.pth
```

Train CCAM on CUB-200-2011 dataset (supervised parameters) 

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
|   |   |   ├—— test
|   |   ├—— ...
```

We also provide the extracted class-agnositc bounding boxes at [here]().

### Reference

If you are using our code, please consider citing our paper.

```
@InProceedings{CCAM,
    author    = {Xie, Jinheng and Xiang, Jianfeng and Chen, Junliang and Hou, Xianxu and Zhao, Xiaodong and Shen, Linlin},
    title     = {CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation},
    booktitle = {CVPR},
    year      = {2022},
}
```

