import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import cv2
import random


def loader(path):
    return cv2.imread(path)


torch.manual_seed(2)

train_datasets = datasets.ImageFolder(root='data/folder/101_ObjectCategories',
                                      loader=loader)  # 标签为{'cats':0, 'dogs':1}

train_size = int(0.2 * len(train_datasets))
test_size = train_size
lefttt = len(train_datasets) - train_size - test_size
train_dataset, test_dataset, _= torch.utils.data.random_split(
    train_datasets, [train_size, test_size, lefttt])

trainset = train_dataset
testset = test_dataset
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1,
                                          shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
