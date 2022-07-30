### Refine background cues

Download the following pre-trained models [GoogleDrive](https://drive.google.com/drive/folders/1Q2Fg2KZV8AzNdWNjNgcavffKJBChdBgy) | [BaiduYun (pwd: 27p5)](https://pan.baidu.com/share/init?surl=ehZheaqeU3pyvYQfRU9c6A) into dataset/pretrained folder.

1. Adopt the extracted background cues as supervision signal to train PoolNet:
```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python3 main_voc.py --arch resnet --mode train --train_root /path/to/your/dataset/VOC2012/ --pseudo_root ../experiments/predictions/path/to/your/background/cues/
```
the models and visual results will be saved as follow: 
```
├── results/
|   ├── run-xx
|   |   ├—— models
|   |   |   ├—— xxx.pth
├── visual_results
```
2. Generate the refined background cues using PoolNet

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python3 main_voc.py --arch resnet --mode test --train_root /path/to/your/dataset/VOC2012/ --model ./results/path/to/your/saved/model --sal_folder ./results/refined_background_cues
```

The refined background cues will be saved as follow:

```
├── results/
|   ├── refined_background_cues
|   |   ├—— xxxx.png
```

**Please Note that** we just follow PoolNet train a saliency detector so the generated results actually activate the foreground (white). **(1 - results/255)** to obtain the real background cues.
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

This repository was modified from [PoolNet](https://github.com/backseason/PoolNet).

