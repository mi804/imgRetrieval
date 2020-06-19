import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from model import OneObjectClassifier
import torch.nn as nn
import time
import mmcv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = models.vgg19(pretrained=True)
        self.features = model.features
        # for p in self.parameters():
        #     p.requires_grad = False
        self.binary = nn.Sequential(nn.Linear(25088, 4096, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5, inplace=False),
                                    nn.Linear(4096, 4096, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(4096, 128, bias=True),
                                    nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(128, 102, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.binary(x)
        x = self.classifier(x)
        return x

    def get_bianry_code(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.binary(x)
        binary = (x > 0.5)
        return binary
