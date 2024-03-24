"""
Contains the training loop for the dual mode paper
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import mnistdata
from src.models import targetmodel, Autoencoder
from src.helper_func import accuracy

# from summary import summary
from torchsummary import summary

## argument parser
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--dataset_dir', type=str, default='/raid/infolab/suma/gm/dataset', help='directory to save the dataset')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 50)')
parser.add_argument('--lr',type = float,default=1e-3,help='learning rate')

args = parser.parse_args()



model = Autoencoder()


print(model)

model.forward(torch.randn(1,1,28,28))

print("------------------------")

