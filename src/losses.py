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
    """
    Implementation of loss function for the targeted adversarial attack generation
    $ l(x) = max(Z(x_i) : i != t) - Z(x)_t $ where Z(x) is the model output and t is the target class
    """
    def __init__(self):
        super(attackLossTar, self).__init__()

    def forward(self, output, targetC):
        """
        output: model output, batch_size x num_classes
        targetC: target class (or classes?) batch_size x 1 or let it be batchsize?
        """
        with torch.no_grad():   ## no need to include the computation of mask in the computation graph (ig)
            mask = torch.ones_like(output, dtype=torch.bool)
            mask[torch.arange(targetC.shape[0]), targetC] = 0
        print(mask.shape)
        maxother = torch.max(output*mask, dim=1)[0]
        print(maxother.shape)
        print(output[torch.arange(targetC.shape[0]),targetC].shape)
        return maxother - output[torch.arange(targetC.shape[0]),targetC]


class attackLossUntar(nn.Module):
    """
    Implementation of loss function for the untargeted adversarial attack generation
    $ l(x) = Z(x)_t - max(Z(x_i) : i != t) $ where Z(x) is the model output and t is the original class
    """
    def __init__(self):
        super(attackLossUntar, self).__init__()

    def forward(self, output, targetC):
        """
        output: model output, batch_size x num_classes
        targetC: target class (or classes?) batch_size x 1
        """
        with torch.no_grad():
            mask = torch.ones_like(output, dtype=torch.bool)
            mask[torch.arange(targetC.shape[0]), targetC] = 0
        maxother = torch.max(output*mask, dim=1)[0]
        return output[torch.arange(targetC.shape[0]),targetC] - maxother