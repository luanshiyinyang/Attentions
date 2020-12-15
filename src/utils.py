import os

import yaml

from models.resnet.model import resnet50
from models.senet.model import senet50
from models.cbam.model import cbam


def read_config(config_file="../config/config.yaml"):
    current_path = os.path.dirname(__file__)
    config_file = os.path.join(current_path, config_file)
    assert os.path.isfile(config_file), "not a config file"
    with open(config_file, 'r', encoding="utf8") as f:
        cfg = yaml.safe_load(f.read())

    return cfg


class Config(object):

    def __init__(self):
        self.config = read_config()
        # data
        data_info = self.config['data']
        self.dataset_name = data_info['dataset']
        dataset_info = data_info[self.dataset_name]
        self.root_folder = dataset_info['root_folder']
        self.txt_path = dataset_info['txt_path']
        self.split = list(map(int, data_info['split'].split(':')))
        self.split = [x / sum(self.split) for x in self.split]
        self.n_classes = dataset_info['classes']
        # train
        train_info = self.config['train']
        self.model = train_info['model']
        self.img_size = train_info['img_size']
        self.lr = train_info['lr']
        self.epochs = train_info['epochs']
        self.batch_size = train_info['bs']
        self.device = train_info['device']
        # exp
        exp_info = self.config['exp']
        self.exp_name = os.path.join(exp_info['dir'], exp_info['name'])

    def __repr__(self):
        return str(self.config)


def get_model(name):
    cfg = Config()
    model_dict = {
        'resnet': resnet50,
        'senet': senet50,
        'cbam': cbam
    }
    model = model_dict[name](n_classes=cfg.n_classes)
    return model


def reformat_tiny_ds():
    import glob
    import os
    from shutil import move
    from os import rmdir

    target_folder = '/home/zhouchen/Datasets/tiny-imagenet-200/val/'

    val_dict = {}
    with open(target_folder + 'val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(target_folder + 'images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/' + str(file)
        move(path, dest)

    os.remove('/home/zhouchen/Datasets/tiny-imagenet-200/val/val_annotations.txt')
    rmdir('/home/zhouchen/Datasets/tiny-imagenet-200/val/images')


if __name__ == '__main__':
    reformat_tiny_ds()