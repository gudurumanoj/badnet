"""
contains code for torch models used in the project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class targetmodel(nn.Module):
    """
    Target model as per the paper
    """
    def __init__(self):
        super(targetmodel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        ## hardcoding output shapes for mnist dataset
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 200), 
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)  # Softmax layer with input size of (200)
        )

        # self.softmax = F.softmax()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten before passing to fully connected layers
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)


## using autoencoder for learning noise
"""
Contains the autoencoder model
Increase complexity and add a latent dimension FC layer if training is not good
"""
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  

        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)

        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##

        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output

        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)


        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        print("----Before conv1------", x.shape)
        x = F.relu(self.conv1(x))
        print("----After conv1------", x.shape)
        x = self.pool(x)
        print("----After pool------", x.shape)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        print("----After conv2------", x.shape)
        x = self.pool(x)
        print("----After pool------", x.shape)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        print("----After conv3------", x.shape)
        x = self.pool(x)  # compressed representation
        print("----After pool------", x.shape)
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        print("----After t_conv1------", x.shape)
        x = F.relu(self.t_conv2(x))
        print("----After t_conv2------", x.shape)
        x = F.relu(self.t_conv3(x))
        print("----After t_conv3------", x.shape)
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
        print("----After conv_out------", x.shape)
                
        return x