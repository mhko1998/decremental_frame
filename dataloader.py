import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import utils

class ImageDataLoader(Dataset):
    
    def __init__(self,dir,images,transform):
        self.images=images
        self.transform=transform
        self.dir=dir
        self.label_dict, self.name_dict=self.__labeling__()

    def __labeling__(self):
        dirname=self.dir
        label_dict=dict()
        name_dict=dict()
        i = [i for i in range(len(dirname))]
        for y in range(len(dirname)):
            label=dirname[y].split('/')[-1]
            label_dict[label]=i[y]
            name_dict[i[y]]=label
        return label_dict, name_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgname=self.images[index]
        x=imgname.split('/')[-2]
        label=self.label_dict[x]
        image=Image.open(imgname)
        image=self.transform(image)
        return image, label

class ReduceImageDataLoader(ImageDataLoader):
    def __init__(self,dir,images,transform,numclass):
        super().__init__(dir,images,transform)
        for i in enumerate(images):
            x=images[i].split('/')[-2]
            label_dict,_=super().__labeling__()
            label=label_dict[x]
            if label<self.numclass:
                self.image=images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        imgname=self.images[index]
        x=imgname.split('/')[-2]
        label_dict, _=super().__labeling__()
        label=label_dict[x]
        image=Image.open(imgname)
        image=self.transform(image)
        return image, label
     

def data_loader():
    config=utils.read_conf('/home/minhwan/cifar100/conf.json')
    stats=((0.507,0.487,0.441),(0.267,0.256,0.276))
    train_trans=transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(*stats,inplace=True)
                            ])
    valid_trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(*stats)])

    trainimages=glob.glob(config["dataset"]+'TRAIN/*/*.png')
    traindir=glob.glob(config["dataset"]+'TRAIN/*')

    testimages=glob.glob(config["dataset"]+'TEST/*/*.png')
    testdir=glob.glob(config["dataset"]+'TEST/*')

    trainset=ImageDataLoader(traindir,trainimages,train_trans)
    testset=ReduceImageDataLoader(testdir,testimages,valid_trans,20)
    

    batch_size=int(config["batch_size"])

    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=8)
    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=8)

    return trainloader, testloader

if __name__=="__main__":
    a,b=data_loader()
    print(len(a),len(b))