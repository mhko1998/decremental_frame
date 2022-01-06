import torch
from torch._C import device
import getnet
import dataloader
import training
import torch.nn as nn

def testing():
    device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    net=getnet.Net()
    net.to(device)
    net.eval()
    net1=getnet.reduceNet()
    net.load_state_dict(torch.load("/home/minhwan/cifar100/modelstate/cifar-resnet34-185.pt"),strict=False)

    for key in net.state_dict():
        if "fc" not in key:
            net1.state_dict()[key]=net.state_dict()[key]
        else:
            for i in range(20):
                net1.state_dict()[key][i]=net.state_dict()[key][i]

    _, testloader = dataloader.data_loader()
    net1.to(device)
    criterion = nn.CrossEntropyLoss()

    training.test(net1,testloader,criterion,device)

if __name__=='__main__':
    testing()