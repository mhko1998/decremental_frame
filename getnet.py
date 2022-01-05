import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
class Net(nn.Module):
    def __init__(self):
        config=utils.read_conf("/home/minhwan/cifar100/conf.json")
        super().__init__()
        self.network=torchvision.models.resnet34(pretrained=False)
        self.network.conv1 =  nn.Conv2d(3, 64, 3, 1)
        self.network.maxpool =  nn.MaxPool2d(1,1)
        num_ftrs=self.network.fc.in_features
        self.network.fc=nn.Linear(num_ftrs,int(config["num_classes"]))
    def forward(self,xb):
        return self.network(xb)