import torch
from torch._C import device
import getnet
import dataloader
import training
import torch.nn as nn
import os.path

def testing():
    device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    net=getnet.Net()
    net.to(device)
    net.eval()
    net1=getnet.reduceNet()
    net.load_state_dict(torch.load("/home/minhwan/cifar100/modelstate/cifar-resnet34-185.pt"),strict=False)
    # for key in net.state_dict():
    #     print(key)
    # for key in net1.state_dict():
    #     print(key)
    for key in net.state_dict():
        # print(key)
        if "fc" not in key:
            net1.state_dict()[key]=net.state_dict()[key]
        else:
            net1.state_dict()[key]=net.state_dict()[key][:20]
    
    
    
    # print(net1.state_dict()["network.conv1.weight"])
    # print(net.state_dict()["network.conv1.weight"])

    _, testloader = dataloader.data_loader()
    net1.to(device)
    criterion = nn.CrossEntropyLoss()

    # training.test(net1,testloader,criterion,device)

if __name__=='__main__':
    testing()