import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torchvision.models.resnet34(pretrained=False)
        self.network.conv1 =  nn.Conv2d(3, 64, 3, 1)
        self.network.maxpool =  nn.MaxPool2d(1,1)
        num_ftrs=self.network.fc.in_features
        self.network.fc=nn.Linear(num_ftrs,100)
    def forward(self,xb):
        return self.network(xb)