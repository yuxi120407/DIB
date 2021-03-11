# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 21:12:00 2021

@author: Xi Yu, Shujian Yu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import argparse

from torchvision import datasets, transforms
from pytorch_model_summary import summary

from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import argparse

from model import VGG
from utils import calculate_MI
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='DIB')
parser.add_argument(
        '--lr',
        type=float, 
         default=0.1, 
        help='learning rate')

parser.add_argument(
   "--epochs",
   type=int,
   default=300,
   help="Number of training epochs. Default: 300")

parser.add_argument(
   "--weight-decay",
   "-wd",
   type=float,
   default=5e-4, #5e-4
   help="L2 regularization factor. Default: 2e-4")

args = parser.parse_args()


#--------------------CIFAR-10----------------------------------#
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=6)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#--------------------Model----------------------------------#

print('==> Building model..')

net = VGG("VGG16")
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

#--------------------Train and Test----------------------------------#
def train(epoch):
    #print('\nEpoch: %d' % epoch)
    print('\nEpoch [{}/{}]'.format(epoch+1, args.epochs))
    net.train()
    train_loss = 0
    IXZ_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        Z,outputs = net(inputs)
        loss = criterion(outputs, targets)
        with torch.no_grad():
            Z_numpy = Z.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) 
        IXZ = calculate_MI(inputs,Z,s_x=1000,s_y=sigma**2)
        total_loss = loss + 0.01*IXZ
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        IXZ_loss += IXZ.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print ('Step [{}/{}], Loss: {:.4f},I_xz: {:.4f},  Acc: {}% [{}/{}])' 
               .format(batch_idx, 
                       len(trainloader), 
                       train_loss/(batch_idx+1),
                       IXZ_loss/(batch_idx+1),
                       100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {}% [{}/{}])' 
                   .format(batch_idx, 
                           len(testloader), 
                           test_loss/(batch_idx+1),
                           100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_IB.pth')
        best_acc = acc
    return acc


#--------------------Main----------------------------------#
best_acc = 0
all_IB_acc = []
for epoch in range(args.epochs):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    all_IB_acc.append(acc)
