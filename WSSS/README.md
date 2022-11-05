### For WSSS task

Download the pretrained parameters (e.g., moco and detco) at [here](https://drive.google.com/drive/folders/1erzARKq9g02-3pUGhY6-hyGzD-hoty5b?usp=sharing) and put them in the current directory.

```
├── WSSS/
|   ├── core
|   ├—— ...
|   ├—— moco_r50_v2-e3b0c442.pth
|   └── detco_200ep.pth
```

1. Train CCAM on PASCAL VOC2012 dataset (unsupervised parameters). **Please ensure the batch size is larger than 32. You can specify more GPUs to allocate enough memory.**

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_VOC12.py --tag CCAM_VOC12_MOCO --batch_size 128 --pretrained mocov2 --alpha 0.25
```

or

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_VOC12.py --tag CCAM_VOC12_MOCO --batch_size 128 --pretrained detco --alpha 0.25
```

We recommend to adopt a batch size 128 for better performance, but you can try another one like 32 if the memory of your device is not enough. **(We trained CCAM on Tesla A100 with 40GB memory.)**

The code will create experiment folders for model checkpoints (./experiment/models), log files (.experiments/logs) and visualization (./experiments/images/).

```
├── experiments/
|   ├── checkpoints
|   ├—— images
```

2. To extract class-agnostic activation maps

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python inference_CCAM.py --tag CCAM_VOC12_MOCO --domain train
```

The activation maps (visualization and .npy files) will be saved at 

```
├── experiments/
|   ├── predictions
|   |   ├—— CCAM_CUB_MOCO@train@scale=0.5,1.0,1.5,2.0
├── vis_cam/
|   ├── CCAM_CUB_MOCO@train@scale=0.5,1.0,1.5,2.0
```

3. To extract background cues

```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python inference_crf.py --experiment_name CCAM_VOC12_MOCO@train@scale=0.5,1.0,1.5,2.0 --threshold 0.3 --crf_iteration 10 --domain train
```

The background cues will be saved at 

```
├── experiments/
|   ├── predictions
|   |   ├—— CCAM_VOC12_MOCOV2@train@scale=0.5,1.0,1.5,2.0@t=0.3@ccam_inference_crf=10
```

4. Refine the background cues
You can use the extracted background cues as pseudo supervision signal to train a saliency detector like [PoolNet](https://github.com/backseason/PoolNet) to further refine the background cues and we provide the code for background cues refinement in the directory `./PoolNet`. We also provide our refined background cues at [here](https://drive.google.com/drive/folders/1erzARKq9g02-3pUGhY6-hyGzD-hoty5b).

5. CAMs Refinement
```shell
python3 evaluate.py --experiment_name you_experiment_name --domain train --data_dir path/to/your/data --with_bg_cues True --bg_dir path/to/your/background cues
```

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

This repository was based on PuzzleCAM and thanks for [Sanghyun Jo](https://github.com/OFRIN/PuzzleCAM) providing great codes.

