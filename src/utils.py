import os

import yaml

from models.resnet.model import resnet50
from models.senet.model import se_resnet50


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
        dataset_name = data_info['dataset']
        dataset_info = data_info[dataset_name]
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
        'senet': se_resnet50
    }
    model = model_dict[name](n_classes=cfg.n_classes)
    return model