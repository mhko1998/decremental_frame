from os import confstr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
from ptflops import get_model_complexity_info
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

class reduceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torchvision.models.resnet34(pretrained=False)
        self.network.conv1 =  nn.Conv2d(3, 64, 3, 1)
        self.network.maxpool =  nn.MaxPool2d(1,1)
        num_ftrs=self.network.fc.in_features
        self.network.fc=nn.Linear(num_ftrs,20)
    def forward(self,xb):
        return self.network(xb)

class decreaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torchvision.models.resnet34(pretrained=False)
        self.network.conv1 =  nn.Conv2d(3, 64, 3, 1)
        self.network.maxpool =  nn.MaxPool2d(1,1)
        num_ftrs=self.network.fc.in_features
        self.network.fc=nn.Linear(num_ftrs,80)
        self.fc=nn.Linear(80,20)
    def forward(self,xb):
        x=self.network(xb)
        output = x
        x=self.fc(x)
        return x

if __name__=="__main__":
    model=Net()
    print(model)
    model1=reduceNet()
    print(model1)
    model2=decreaseNet()
    print(model2)