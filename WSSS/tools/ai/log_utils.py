# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
from tools.general.txt_utils import add_txt

def log_print(message, path):
    """This function shows message and saves message.
    
    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.
        
        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
    print(message)
    add_txt(path, message)

class Logger:
    def __init__(self):
        pass

class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys
        
        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]
            
        return dataset
    
    def clear(self):
        self.data_dic = {key : [] for key in self.keys}

