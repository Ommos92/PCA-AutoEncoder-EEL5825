from __future__ import print_function
import argparse
import os
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from convolutional_autoencoder import Autoencoder 
import argparse
import numpy as np 
import utils

def train(model, trainloader, epochs, device, criterion, optimizer):
    #initialize training loss
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        #Perform operation on each dataset in epoch
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)

        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, epochs, loss))

        if epoch % 1 == 0:
            img = img.view(img.size(0),1,28,28)
            utils.save_image(img, './MNIST_Images/MNIST_Original_%i.png' % epoch)
            utils.save_decoded(outputs.cpu().data, epoch)

    return train_loss
    
def test(model, test_loader, device, criterion, optimizer):
    for batch in test_loader:
        img, _ = batch 
        img = img.to(device)
        img = img.view(img.size(0), -1)
        utils.save_image(img, './Mnist_Autoencoder/MNIST_Reconstructed/MNIST_Original.png')
        outputs = model(img)
        outputs = outputs.view(outputs.size(0),1,28,28)
        utils.save_image(outputs, './Mnist_Autoencoder/MNIST_Reconstructed/MNIST_Reconstruction.png')
        break
    