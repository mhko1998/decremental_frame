import neptune
from neptune.new import run
import torch
from tqdm import tqdm
import getoptim

def train(net,trainloader,optimizer,criterion,device,epoch):
    
    net.train()
    running_loss=0.0
    total=0
    correct=0
    for i, data in tqdm(enumerate(trainloader,0)):
        image, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs=net(image)
        loss=criterion(outputs, label)
        loss.backward()
        optimizer.step()
        _,predicted=torch.max(outputs.data,1)
        total+=label.size(0)
        correct+=(predicted==label).sum().item()
        running_loss += loss.item()
    train_acc=correct*100/total
    print('[%d] loss: %.3f'  % (epoch + 1, running_loss / len(trainloader)))
    return running_loss , train_acc

def test(net, testloader,criterion, device):
    net.eval()
    correct = 0
    total = 0
    valloss=0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader,0)):
            image, label = data[0].to(device), data[1].to(device)
            outputs = net(image)
            loss=criterion(outputs,label)
            valloss+=loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    valacc=correct*100/total
    print('Accuracy of the network on the test images: %d %%' %(100*correct/total))
    print(correct, total)
    
    return valloss, valacc
