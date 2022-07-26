### Train a regressor using the psuedo bboxes extracted from CCAM
#### CUB-200-2011, plase specify the path to dataset and pseudo bboxes and type of backbone in `run_cub.sh`. For convenience, we provide the pretrained classifier (efficientnetb7) on CUB at [here](https://drive.google.com/file/d/1FvtrT-TybVDBkYmRpWeAU4i0l3xLo10V/view?usp=sharing) and put it on the current directory.

```
bash run_cub.sh
```
#### ILSVRC2012, plase specify the path to dataset and pseudo bboxes and type of backbone in `run_ilsvrc.sh`. Download the gt bboxes for validation at [here](https://drive.google.com/file/d/1D501nKfJ2TZGoHqj0neNL93G4R8Knd4b/view?usp=sharing) and put it on the current directory.

```
bash run_ilsvrc.sh
```