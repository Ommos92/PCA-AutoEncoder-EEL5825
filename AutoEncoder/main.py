from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from convolutional_autoencoder import Autoencoder
import train_evaluate_autoencoder
import utils
import argparse
import numpy as np 
    
def run_main(FLAGS):
    # Check if cuda is available
    device = utils.get_device()
    
    print("Device: {}".format(device))

    #Make Directory for storing images
    utils.dir()

    # Initialize the model and send to device 
    model = Autoencoder(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    # ======================================================================
    criterion = nn.MSELoss()
    
    # ======================================================================
    # Define optimizer function.
    # ======================================================================
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.CIFAR10('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train_evaluate_autoencoder.train(model, train_loader, FLAGS.num_epochs, device, criterion, optimizer)
        
        test_loss = train_evaluate_autoencoder.test(model, test_loader, device, criterion, optimizer)
        
        #if test_accuracy > best_accuracy:
        #    best_accuracy = test_accuracy
    
    
    #print("accuracy is {:2.2f}".format(best_accuracy))
    
    #print("Training and evaluation finished")
    return train_loader, test_loader
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    