"""
contains code for different loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class l2normLoss(nn.Module):
    """
    L2 norm loss
    """
    def __init__(self):
        super(l2normLoss, self).__init__()

    def forward(self, xOrg, xAdv):
        return torch.mean(torch.norm(xOrg - xAdv, p=2, dim=1))
    

class l1normLoss(nn.Module):
    """
    L1 norm loss
    """
    def __init__(self):
        super(l1normLoss, self).__init__()

    def forward(self, xOrg, xAdv):
        return torch.mean(torch.norm(xOrg - xAdv, p=1, dim=1))
    
class attackLossTar(nn.Module):
    def __init__(self):
        super(attackLossTar, self).__init__()

    def forward(self, output, target):
        pass


class attackLossUntar(nn.Module):
    def __init__(self):
        super(attackLossUntar, self).__init__()

    def forward(self, output, target):
        pass