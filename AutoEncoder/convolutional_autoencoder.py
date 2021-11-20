# import packages
import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Autoencoder(nn.Module):
    """
    Autoencoder Model
    """
    def __init__(self, mode):
        super(Autoencoder,self).__init__()


        
        # encoders
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
       
        # decoders
        self.dec1 = nn.Linear(in_features=128, out_features=256)
        self.dec2 = nn.Linear(in_features=256, out_features=784)

        #Convolutional Autoencoder

        #Encoders
        self.conv1 = nn.Conv2d(1,24,3)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(24,48,3)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2)
        self.reLU = nn.ReLU()
        
        #Decoders
        self.convTranspose1 = nn.ConvTranspose2d(in_channels=48, out_channels=24,kernel_size=3)
        self.upSample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convTranspose2 = nn.ConvTranspose2d(in_channels=24, out_channels=8,kernel_size=3)
        self.upSample2 = nn.UpsamplingBilinear2d(scale_factor=1.75)
        self.convTranspose3 = nn.ConvTranspose2d(in_channels=8, out_channels=1,kernel_size=1)
        

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2

            
    # Baseline model. step 1
    def model_1(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
       
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
       
        return x

    # Use two convolutional layers.
    def model_2(self, x):
       #Return to original dimensions
       
       #Encoder Stage
       x = x.view(10,1,28,28)
       x = self.conv1(x)
       x = self.MaxPool1(x)
       x = self.conv2(x)
       x = self.MaxPool2(x)
       x = self.reLU(x)
       #print(x.shape)
       #Decoder Stage
       x = self.convTranspose1(x)
       x = self.upSample1(x)
       x = self.convTranspose2(x)
       x = self.upSample2(x)
       x = self.convTranspose3(x)
       x = x.view(10,784)
       return x 
