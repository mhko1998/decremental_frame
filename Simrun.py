import torch
from torch import optim
import dataloader
import getnet
import getoptim
import getscheduler
import torch.nn as nn
import torchvision
import training
import os
import neptune.new as neptune


def Simrun():
    maxacc=0
    current_epoch=0
    
    device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader = dataloader.data_loader()

    net = getnet.Net()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = getoptim.getoptim(net)

    scheduler= getscheduler.getscheduler(optimizer)
    run = neptune.init(
    project="mhko1998/cifar100resnet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOGQ5Y2U4OC0xZWIzLTQyZjQtYWIyMy0wNTA5N2ExMzg2N2IifQ==",)
    for epoch in range(201):
        trainloss, trainacc=training.train(net,trainloader,optimizer,criterion,device,epoch)
        run['train/loss'].log(trainloss)
        run['train/acc'].log(trainacc)
        scheduler.step()

        if epoch % 5 == 0:
            valloss,valacc=training.test(net,testloader,device)
            run['val/loss'].log(valloss)
            run['val/acc'].log(valacc)
            if valacc > maxacc:
                if os.path.exists('/home/minhwan/cifar100/modelstate/cifar-resnet34-'+str(current_epoch)+'.pt'):
                    os.remove('/home/minhwan/cifar100/modelstate/cifar-resnet34-'+str(current_epoch)+'.pt')
                torch.save(net.state_dict(),'/home/minhwan/cifar100/modelstate/cifar-resnet34-'+str(epoch)+'.pt')
                current_epoch=epoch

        
    print('Fininshed Training')

if __name__=='__main__':
    Simrun()