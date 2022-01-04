import torch

def getscheduler(optim):
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 120, 160],gamma=0.2)
    return train_scheduler