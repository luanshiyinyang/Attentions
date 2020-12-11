import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from runx.logx import logx

from utils import Config, get_model
from dataset import Caltech, get_tfms

config = Config()
logdir = config.exp_name
logx.initialize(logdir, coolname=True, tensorboard=True)


def train_epoch(epoch):
    model.train()
    losses = 0.0
    total, correct = 0.0, 0.0
    for step, (x, y) in enumerate(train_loader):
        with autocast():
            x, y = x.to(config.device), y.to(config.device)
            out = model(x)
            loss = criterion(out, y)
        losses += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, pred = torch.max(out.data, 1)
        total += y.size(0)
        correct += (pred == y).squeeze().sum().cpu().numpy()

        if step % 100 == 0:
            logx.msg("epoch {} step {} training loss {}".format(epoch, step, loss.item()))
    logx.msg("epoch {} training loss {} training acc {}".format(epoch, losses / (step + 1), correct / total))
    logx.metric("train", {"loss": losses / (step + 1), 'acc': correct / total})
    return losses


def test_epoch(epoch):
    model.eval()
    losses = 0.0
    total, correct = 0.0, 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(val_loader):
            x, y = x.to(config.device), y.to(config.device)
            out = model(x)
            loss = criterion(out, y)
            losses += loss.cpu().detach().numpy()
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y).squeeze().sum().cpu().numpy()
    save_dict = {
        'state_dict': model.state_dict()
    }
    logx.msg("epoch {} validation loss {} validation acc {}".format(epoch, losses / (step + 1), correct / total))
    logx.metric('val', {'loss': losses / (step + 1), 'acc': correct / total})
    logx.save_model(save_dict, losses, epoch, higher_better=False, delete_old=True)


# dataset

tfms = get_tfms(config.img_size)
train_ds = Caltech(txt=os.path.join(config.txt_path, 'train.txt'), transform=tfms)
val_ds = Caltech(txt=os.path.join(config.txt_path, 'train.txt'), transform=tfms)
train_loader = DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=config.batch_size, shuffle=False)
print("data load successfully")
# model
model = get_model(config.model)
model = model.to(config.device)
scaler = GradScaler()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

for i in range(config.epochs):
    train_epoch(i)
    test_epoch(i)