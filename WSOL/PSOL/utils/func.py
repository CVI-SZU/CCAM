import copy

import numpy as np
import torch


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item == 0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
    return count_dict[0][0]


def sk_pca(X, k):
    from sklearn.decomposition import PCA
    pca = PCA(k)
    pca.fit(X)
    vec = pca.components_
    # print(vec.shape)
    return vec


def fld(x1, x2):
    x1, x2 = np.mat(x1), np.mat(x2)
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    k = x1.shape[1]

    m1 = np.mean(x1, axis=0)
    m2 = np.mean(x2, axis=0)
    m = np.mean(np.concatenate((x1, x2), axis=0), axis=0)
    print(x1.shape, m1.shape)

    c1 = np.cov(x1.T)
    s1 = c1 * (n1 - 1)
    c2 = np.cov(x2.T)
    s2 = c2 * (n2 - 1)
    Sw = s1 / n1 + s2 / n2
    print(Sw.shape)
    W = np.dot(np.linalg.inv(Sw), (m1 - m2).T)
    print(W.shape)
    W = W / np.linalg.norm(W, 2)
    return np.mean(np.dot(x1, W)), np.mean(np.dot(x2, W)), W


def pca(X, k):
    n, m = X.shape
    mean = np.mean(X, 0)
    # print(mean.shape)
    temp = X - mean
    conv = np.cov(X.T)
    # print(conv.shape)
    conv1 = np.cov(temp.T)
    # print(conv-conv1)

    w, v = np.linalg.eig(conv)
    # print(w.shape)
    # print(v.shape)
    index = np.argsort(-w)
    vec = np.matrix(v.T[index[:k]])
    # print(vec.shape)

    recon = (temp * vec.T) * vec + mean

    # print(X-recon)
    return vec


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x)


def to_data(x):
    if torch.cuda.is_available():
        # print(x.device)
        x = x.cpu()
        # x = x.to('cpu')
    return x.data


def copy_parameters(model, pretrained_dict):
    model_dict = model.state_dict()

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                       k[7:] in model_dict and pretrained_dict[k].size() == model_dict[k[7:]].size()}
    # for k, v in pretrained_dict.items():
    # print(k)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def compute_intersec(i, j, h, w, bbox):
    '''
    intersection box between croped box and GT BBox
    '''
    intersec = copy.deepcopy(bbox)

    intersec[0] = max(j, bbox[0])
    intersec[1] = max(i, bbox[1])
    intersec[2] = min(j + w, bbox[2])
    intersec[3] = min(i + h, bbox[3])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''

    intersec[0] = (intersec[0] - j) / w
    intersec[2] = (intersec[2] - j) / w
    intersec[1] = (intersec[1] - i) / h
    intersec[3] = (intersec[3] - i) / h
    return intersec
