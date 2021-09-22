import numpy as np


def func(x):
    # return sqrt(x)
    return np.sqrt(np.sin(x) / np.cos(x))


def bench_dataset(el, lst):
    lst[0] = el
