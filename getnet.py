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

if __name__=="__main__":
    model=Net()
    dummy_size=(3,256,256)
    macs, params= get_model_complexity_info(model, dummy_size,as_strings=True,
    print_per_layer_stat=True,verbose=True)
    print('computational complexity: ', macs)
    print('number of parameters: ',params)
    