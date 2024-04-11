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
from src.helper_func import accuracy, accuracy2
from torch.cuda.amp import GradScaler, autocast



## argument parser
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--dataset_dir', type=str, default='/raid/infolab/suma/gm/dataset', help='directory to save the dataset')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 50)')
parser.add_argument('--lr',type = float,default=1e-4,help='learning rate')
parser.add_argument('--mode', type=int, default=0, help='mode of the attack')
parser.add_argument('--type', type=str, default='untargeted', help='type of the attack')
parser.add_argument('--modelpath', type=str, default='/raid/infolab/suma/gm/models/target_model/model_highest.ckpt', help='path to the model')
parser.add_argument('--save_model', type=str, default='/raid/infolab/suma/gm/models/advImgGen', help='directory to save the model')

args = parser.parse_args()

def get_data_loader(args):
    """
    Get the data loader
    """
    train_loader, test_loader = mnistdata(batch_size=args.batch_size, data_dir=args.dataset_dir)
    return train_loader, test_loader


def loadmodel(path):
    """
    To load the trained mnist model from the saved model
    """
    model = targetmodel()
    model.load_state_dict(torch.load(path))
    return model


def test(advImgGen, model, test_loader, args):
    model.eval()
    advImgGen.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            ## modifying the target to get the test attack accuracy
            if args.type == 'targeted':
                target = (target + 1)%10

            data, target = data.cuda(), target.cuda()

            advImg = advImgGen(data)

            output = model(advImg)

            predictions.append(output.argmax(dim=1))
            targets.append(target)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    if args.type == 'targeted':
        acc = accuracy(predictions, targets)
    else:
        acc = accuracy2(predictions, targets)
    # print("Test Accuracy: ", acc)
    return acc

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
    # scaler = GradScaler()

    ## loss functions
    distortloss = l2normLoss()
    attackloss = attackLossTar() if args.type == 'targeted' else attackLossUntar()

    ## c and g
    g = args.mode ## 0 or 1
    c = 1.0       ## constant weight used in the combination of the loss function


    lossLog = []
    accuracies = []

    bestacc = 0

    for epoch in range(args.epochs):
        loss_epoch = []
        advImgGen.train()
        for i, (data, target) in enumerate(train_loader):
            
            ## target modification as per mentioned in the paper
            if args.type == 'targeted':
                target = (target + 1)%10

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            with autocast():
                advImg = advImgGen(data)
                output = targetModel(advImg)
                # print(output.shape, target.shape)
            
            loss = g * distortloss(data, advImg) + c * attackloss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_epoch.append(loss.item())
            if i % 100 == 0:
                print("Epoch: {}, Iteration: {}, LR: {}, Loss: {:.4f}".format(epoch, i, scheduler.get_last_lr()[0], loss.item()))


        lossLog.append(loss_epoch)

        ## testing
        acc = test(advImgGen, targetModel, test_loader, args)
        print("Test Accuracy: ", acc)
        accuracies.append(acc)
        if acc > bestacc:
            bestacc = acc
            torch.save(advImgGen.state_dict(), os.path.join(args.save_model, 'advImgGen_highest.ckpt'))

    ## saving the losses
    np.save(os.path.join(args.save_model, 'losses.npy'), np.array(lossLog))
    print("----------Training done------------")
    print("Best Accuracy: ", bestacc)





def main(args):
    """
    Main function
    """
    train_loader, test_loader = get_data_loader(args)
    print("-------Data loaded--------")
    targetModel = loadmodel(args.modelpath)
    print("-------Model loaded--------")
    advImgGen = Autoencoder()
    print("-------Autoencoder loaded--------")
    train(targetModel, advImgGen, train_loader, test_loader, args)

if __name__ == '__main__':
    main(args)