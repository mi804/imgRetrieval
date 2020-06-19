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


# data_transform = transforms.Compose([
#     transforms.Resize(256),  # 把图片resize为256*256
#     transforms.CenterCrop(224),  # 随机裁剪224*224
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
#                                                           0.225])  # 标准化
# ])
def loader(path):
    return cv2.imread(path)


data_transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.ImageFolder(root='data/folder/102flowers/train',
                                loader=loader)  # 标签为{'cats':0, 'dogs':1}
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1,
                                          shuffle=False)

testset = datasets.ImageFolder(root='data/folder/102flowers/test',
                               loader=loader)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
