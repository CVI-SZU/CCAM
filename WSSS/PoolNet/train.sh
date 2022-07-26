#! /bin/bash

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=5 python main_voc.py --arch resnet --mode train --train_root /data/xjheng/data/VOC2012/ --pseudo_root ../CFC-WSSSv2/experiments/predictions/c2am-voc2012-7@train@scale=0.5,1.0,1.5,2.0@cam_inference_crf=10-soft/
# you can optionly change the -lr and -wd params
