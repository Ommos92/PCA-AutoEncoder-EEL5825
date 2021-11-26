import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#Utilities for Autoencoder Pytorch Implementaiton
def get_device():
    """
    Get GPU or CPU device for framework
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def dir():
    """
    Make directory for storing decoded images
    """
    img_dir = 'MNIST_Images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

def save_decoded(img,epoch):
    img = img.view(img.size(0),3,32,32)
    save_image(img, './MNIST_Images/autoencoded_img_{}.png'.format(epoch))
