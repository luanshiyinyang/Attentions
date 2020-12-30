import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile
from thop import clever_format

from utils import read_tfevents, get_model


def plot_history(ds="caltech256"):
    base_dir = '../runs/'
    exp_dirs = list(filter(lambda x: x.startswith(ds), os.listdir(base_dir)))
    plt.figure(figsize=(12, 6))
    for exp in exp_dirs:
        train_loss, train_acc, val_loss, val_acc = read_tfevents(os.path.join(base_dir, exp))
        if exp.endswith('resnet'):
            plt.plot(np.arange(len(train_loss[:30])), train_loss[:30], label=exp, marker='x')
            # plt.plot(np.arange(len(val_loss[:30])), val_loss[:30], label=exp, marker='o')
        else:
            plt.plot(np.arange(len(train_loss[:30])), train_loss[:30], label=exp)
            # plt.plot(np.arange(len(val_loss[:30])), val_loss[:30], label=exp)
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('training loss')
    plt.show()


def plot_best_acc():
    base_dir = '../runs/'
    exp_dirs = list(filter(lambda x: x.startswith('caltech101'), os.listdir(base_dir)))
    plt.figure(figsize=(12, 6))
    for exp in exp_dirs:
        train_loss, train_acc, val_loss, val_acc = read_tfevents(os.path.join(base_dir, exp))
        print(exp, max(val_acc))


def get_model_config():
    models = ['resnet', 'senet', 'cbam', 'eca-net', 'sknet', 'triplet-attention', 'resnest']
    models = list(map(get_model, models))
    for net in models:
        input = torch.randn(1, 3, 224, 224)
        flops, params = profile(net, inputs=(input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)


if __name__ == '__main__':
    plot_best_acc()