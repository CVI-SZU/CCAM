# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

def read_txt(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_txt(path, data_list):
    with open(path, 'w') as f:
        for data in data_list:
            f.write(data + '\n')

def add_txt(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')