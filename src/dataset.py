import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class Caltech(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class TinyImageNet(Dataset):
    def __init__(self, root, subdir='train', transform=None):
        super(TinyImageNet, self).__init__()

        self.transform = transform
        self.image = []

        self.class_names = sorted(os.listdir(os.path.join(root, 'train')))
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        self.get_ds(subdir, root)

    def get_ds(self, subdir, root):
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, subdir)

        if subdir == 'train':
            for label in self.class_names:
                d = os.path.join(train_dir, label)
                for directory, _, names in os.walk(d):
                    for name in names:
                        filename = os.path.join(directory, name)
                        if filename.endswith('JPEG'):
                            self.image.append((filename, self.names2index[label]))
        elif subdir == 'val':

            with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')
                infos = [info.strip().split('\t')[:2] for info in infos]

                self.image = [(os.path.join(val_dir, 'images', info[0]), self.names2index[info[1]]) for info in infos]
        else:
            raise ValueError("not this sub dataset")

    def __getitem__(self, item):
        path, label = self.image[item]
        with open(path, 'rb') as f:  # rb读取二进制文件
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image)


def get_tfms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform

if __name__ == '__main__':
    from utils import Config
    cfg = Config()
    dataset = TinyImageNet(cfg.root_folder, subdir='val')
    print(len(dataset))
