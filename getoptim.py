import torch
from torch import optim
import dataloader
import getnet
from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler

def getoptim(net):
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    return optimizer
