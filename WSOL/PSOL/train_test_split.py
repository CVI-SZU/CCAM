import numpy as np
import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description='train test split')
parser.add_argument('--data_dir', type=str, required=True, dest='path to your dataset, e.g., xx/xx/CUB_200_2011')

args = parser.parse_args()

image_path = f'{args.data_dir}/images/'
id2name = np.genfromtxt(f'{args.data_dir}/images.txt', dtype=str)
id2train = np.genfromtxt(f'{args.data_dir}/train_test_split.txt', dtype=int)

for id_ in range(id2name.shape[0]):
    # load each variable
    image = os.path.join(image_path, id2name[id_, 1])
    folder_name = id2name[id_, 1].split('/')[0]
    if not os.path.exists(f'{args.data_dir}/test/{folder_name}'):
        os.mkdir(f'{args.data_dir}/test/{folder_name}')

    if not os.path.exists(f'{args.data_dir}/train/{folder_name}'):
        os.mkdir(f'{args.data_dir}/train/{folder_name}')

    # Label starts with 0
    if id2train[id_, 1] == 0:
        copyfile(image, f'{args.data_dir}/test/{id2name[id_, 1]}')

    else:
        copyfile(image, f'{args.data_dir}/train/{id2name[id_, 1]}')
