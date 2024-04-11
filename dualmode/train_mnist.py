"""
Contains the training loop for the model
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import mnistdata
from src.models import targetmodel
from src.helper_func import accuracy
# import pickle

## argument parser
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--dataset_dir', type=str, default='/raid/infolab/suma/gm/dualmode/dataset', help='directory to save the dataset')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
parser.add_argument('--lr',type = float,default=0.1,help='learning rate')
parser.add_argument('--momentum',type = float,default=0.9,help='momentum')
parser.add_argument('--save_model', type=str, default='/raid/infolab/suma/gm/dualmode/models/target_model', help='directory to save the model')

args = parser.parse_args()


def train(model, train_loader, test_loader,args):
    """
    Training loop
    Assuming cuda() is present
    model is already in cuda()
    """
    print("----------------- Training -----------------")
    losses = []
    accs = []
    maxacc = 0

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print("Optimizer: ", optimizer)

    for epoch in range(args.epochs):
        loss_epoch = []
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            # loss = F.cross_entropy(output, target)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            if i % 100 == 0:
                print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss.item()))

        losses.append(loss_epoch)
        acc = test(model, test_loader)
        print("Accuracy: ", acc)
        accs.append(acc)
        if acc > maxacc:
            maxacc = acc
            torch.save(model.state_dict(), os.path.join(args.save_model,'model_highest.ckpt'))

    # print(np.array(losses))
    ## saving losses as .npy file
    np.save(os.path.join(args.save_model, 'losses.npy'), np.array(losses))

    print("----------Training done------------")
    # print(A)

    ## saving the model weights
    # torch.save(model.state_dict(), args.save_model)


def test(model, test_loader):
    """
    Testing on test_loader
    """
    model.eval()

    predictions = []
    true_labels = []

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        predictions.append(pred.cpu().detach())
        true_labels.append(target.cpu().detach())

    predictions = torch.cat(predictions)
    true_labels = torch.cat(true_labels)

    acc = accuracy(predictions, true_labels)
    print("Accuracy on test set: {:.4f}".format(acc))
    return acc


if __name__ == '__main__':
    # print("------ Loading data ------")
    train_loader, test_loader = mnistdata(batch_size=args.batch_size, data_dir=args.dataset_dir)
    print("------ Data loaded ------")
    model = targetmodel().cuda()
    model = model.train()
    print("------ Model loaded ------")
    train(model, train_loader, test_loader,args)