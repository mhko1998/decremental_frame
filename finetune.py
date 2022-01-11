import torch
from torch import optim
from torchvision.models.resnet import ResNet
import dataloader
import getnet
import getoptim
import getscheduler
import torch.nn as nn
import torchvision
import training
import os
import neptune.new as neptune
import utils

def finetune():
    config=utils.read_conf('/home/minhwan/cifar100/conf.json')
    maxacc=0
    current_epoch=0
    
    device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    net=getnet.Net()
    net.to(device)
    net.eval()
    net1=getnet.decreaseNet()
    net.load_state_dict(torch.load("/home/minhwan/cifar100/modelstate/cifar-resnet34-185.pt"))
    
    a=dict()
    for key in net.state_dict():
        if "fc" not in key:
            a[key]=net.state_dict()[key]
        else:
            a[key]=net.state_dict()[key][20:]
    torch.save(a,"/home/minhwan/cifar100/modelstate/cifar-resnet34-18.pt")
    net1.load_state_dict(torch.load("/home/minhwan/cifar100/modelstate/cifar-resnet34-18.pt"),strict=False)
    for child in net1.children():
        if type(child)==ResNet:
            for param in child.parameters():
                param.requires_grad=False
    net1.to(device)
    
    trainloader, testloader = dataloader.data_loader()


    criterion = nn.CrossEntropyLoss()

    optimizer = getoptim.getoptim(net1)

    scheduler= getscheduler.getscheduler(optimizer)
    run = neptune.init(
    project="mhko1998/cifar100resnet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOGQ5Y2U4OC0xZWIzLTQyZjQtYWIyMy0wNTA5N2ExMzg2N2IifQ==",)
    
    for epoch in range(int(config["epoch"])):
        trainloss, trainacc=training.train(net1,trainloader,optimizer,criterion,device,epoch)
        run['train/loss'].log(trainloss)
        run['train/acc'].log(trainacc)
        scheduler.step()

        if epoch % 5 == 0:
            net1.to(device)
            valloss,valacc=training.test(net1,testloader,criterion,device)
            run['val/loss'].log(valloss)
            run['val/acc'].log(valacc)
            if valacc > maxacc:
                if os.path.exists(config["save_path"]+'cifar-resnet34-'+str(current_epoch)+'.pt'):
                    os.remove(config["save_path"]+'cifar-resnet34-'+str(current_epoch)+'.pt')
                torch.save(net.state_dict(),config["save_path"]+'cifar-resnet34-'+str(epoch)+'.pt')
                current_epoch=epoch
    print('Fininshed Training')

if __name__=='__main__':
    finetune()