import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split


def get_transforms(mean,std):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transforms,test_transforms


def compute_mean_std(ds):
    all_images = np.zeros((len(ds),32,32,3))
    for i in range(len(ds)):
        all_images[i] = np.array(ds[i][0])
    r = all_images[:,:,:,0]
    g = all_images[:,:,:,1]
    b = all_images[:,:,:,2]
    means = (r.mean()/255.,g.mean()/255.,b.mean()/255.)
    std = (r.std()/255.,g.std()/255.,b.std()/255.)
    return means, std

def get_mean_std(ds):
    if ds == "cifar100":
        return compute_mean_std(datasets.CIFAR100('../data', train=True, download=True))
    elif ds == "cifar10":
        return compute_mean_std(datasets.CIFAR10('../data', train=True, download=True))
    elif ds == "svhn":
        return compute_mean_std(datasets.SVHN('../data', split='train', download=True))


def build_data_loaders(data_set,shuffle=True,batch_size=64,val_size=.1):
    if data_set == "cifar100":
        num_classes = 100
        cifar_100_mean, cifar_100_std = compute_mean_std(datasets.CIFAR100('../data', train=True, download=True))
        train_transform,test_transform = get_transforms(cifar_100_mean,cifar_100_std)
        # Split up
        train_set = datasets.CIFAR100('../data', train=True, download=True,transform=train_transform)
        val_size = int(val_size*len(train_set))
        train_size = len(train_set) - val_size
        train_ds, val_ds = random_split(train_set, [train_size, val_size])
        # now set loaders
        train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, download=True,transform=test_transform),
            batch_size=batch_size, shuffle=shuffle,pin_memory=True)

    elif data_set == "cifar10":
        num_classes = 10
        cifar_10_mean, cifar_10_std = compute_mean_std(datasets.CIFAR10('../data', train=True, download=True))
        train_transform,test_transform = get_transforms(cifar_10_mean,cifar_10_std)

        train_set = datasets.CIFAR10('../data', train=True, download=True,transform=train_transform)
        val_size = int(val_size*len(train_set))
        train_size = len(train_set) - val_size
        train_ds, val_ds = random_split(train_set, [train_size, val_size])
        # now set loaders
        train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True,transform=test_transform),
            batch_size=batch_size, shuffle=shuffle,pin_memory=True)

    elif data_set == "svhn":
        num_classes = 10
        train_set = datasets.SVHN(root='../data', split='train', download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),)
        val_size = int(val_size*len(train_set))
        train_size = len(train_set) - val_size
        train_ds, val_ds = random_split(train_set, [train_size, val_size])
        # now set loaders
        train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root='../data', split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                ),
            batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return train_loader,val_loader,test_loader, num_classes

























#
