#encoding:utf8
import numpy as np

def gaussian(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x-mean)/std

def scale(x):
    std = np.std(x)
    return x/std
