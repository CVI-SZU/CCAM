import numpy as np

def color_map(N = 256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b, g, r])
    
    return cmap

def get_color_map_dic():
    labels = ['background', 
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']

    # n_classes = 21
    n_classes = len(labels)
    
    h = 20
    w = 500

    color_index_list = [index for index in range(n_classes)]

    cmap = color_map()
    cmap_dic = {label : cmap[color_index] for label, color_index in zip(labels, range(n_classes))}
    cmap_image = np.empty((h * len(labels), w, 3), dtype = np.uint8)
    
    for color_index in color_index_list:
        cmap_image[color_index * h : (color_index + 1) * h, :] = cmap[color_index]
    
    return cmap_dic, cmap_image, labels
