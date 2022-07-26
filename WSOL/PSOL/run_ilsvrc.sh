#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python training_ILSVRC.py --pseudo_bboxes_path /path/to/your/pseudo_boxes --save_path ImageNet_resnet50_uns /path/to/ILSVRC2012
CUDA_VISIBLE_DEVICES=6 python inference_ILSVRC.py --loc_model resnet50 --cls_model efficientnetb7 --ckpt ./ImageNet_resnet50_uns/checkpoint_localization_imagenet_resnet50_19.pth.tar /path/to/ILSVRC2012

