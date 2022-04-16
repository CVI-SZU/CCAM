# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import pickle

def dump_pickle(path, data):
    pickle.dump(data, open(path, 'wb'))

def load_pickle(path):
    return pickle.load(open(path, 'rb'))

