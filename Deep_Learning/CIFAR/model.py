import torch.nn as nn
import torch
from resnet_torch import resnet18


class OneObjectClassifier(nn.Module):
    def __init__(self, binary_bits=48, num_classes=10, pretrained=False):
        super(OneObjectClassifier, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.latent_layer = nn.Linear(256, binary_bits)
        self.fc_layer = nn.Linear(binary_bits, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.extract_feat(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.latent_layer(x))
        x = self.fc_layer(x)
        return x

    def extract_feat(self, img):
        return self.backbone(img)

    def get_bianry_code(self, x):
        x = self.extract_feat(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid((self.latent_layer(x)))
        binary = (x > 0.5)
        # TODO: get binary codes
        return binary
