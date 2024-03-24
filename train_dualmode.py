"""
Contains the training loop for the dual mode paper
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import mnistdata
from src.models import targetmodel, Autoencoder
from src.losses import l2normLoss, l1normLoss, attackLossTar, attackLossUntar
from src.helper_func import accuracy
from torch.cuda.amp import GradScaler, autocast



## argument parser
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--dataset_dir', type=str, default='/raid/infolab/suma/gm/dataset', help='directory to save the dataset')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 50)')
parser.add_argument('--lr',type = float,default=1e-3,help='learning rate')
parser.add_argument('--mode', type=int, default='0 or 1', help='mode of the attack')
parser.add_argument('--type', type=str, default='untargeted', help='type of the attack')
parser.add_argument('--modelpath', type=str, default='/raid/infolab/suma/gm/models/target_model', help='path to the model')

args = parser.parse_args()

def get_data_loader(args):
    """
    Get the data loader
    """
    train_loader = torch.utils.data.DataLoader(mnistdata(args.dataset_dir, train=True), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnistdata(args.dataset_dir, train=False), batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def loadmodel(path):
    """
    To load the trained mnist model from the saved model
    """
    model = targetmodel()
    model.load_state_dict(torch.load(path))
    return model

def train(targetModel, advImgGen, train_loader, test_loader, args):
    """
    Assuming cuda is present
    """
    targetModel = targetModel.cuda()
    advImgGen = advImgGen.cuda()
    targetModel.eval()
    advImgGen.train()

    ## optimizer, scheduler and scaler
    optimizer = optim.Adam(advImgGen.parameters(), lr=args.lr)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    scaler = GradScaler()

    ## loss functions
    distortloss = l2normLoss()
    attackloss = attackLossTar() if args.type == 'targeted' else attackLossUntar()

    ## c and g
    g = args.mode ## 0 or 1
    c = 1.0       ## constant weight used in the combination of the loss function


    lossLog = []

    for epoch in range(args.epochs):
        loss_epoch = []
        for i, (data, target) in enumerate(train_loader):
            
            ## target modification as per mentioned in the paper
            target = target + 1

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            with autocast():
                advImg = advImgGen(data)
                output = targetModel(advImg)
                loss = c * distortloss(data, advImg) + g * attackloss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_epoch.append(loss.item())
            if i % 100 == 0:
                print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss.item()))


        lossLog.append(loss_epoch)




def main(args):
    """
    Main function
    """
    train_loader, test_loader = get_data_loader(args)
    targetModel = loadmodel(args.modelpath)
    advImgGen = Autoencoder()
    train(targetModel, advImgGen, train_loader, test_loader, args)