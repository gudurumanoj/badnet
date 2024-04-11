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
        # with torch.no_grad():
        a = torch.arange(output.size(0)).cuda()
        mask = torch.ones_like(output).cuda()
        mask[a, targetC] = 0
        maxNonTarget = torch.max(output * mask, dim=1)[0]
        return torch.mean(maxNonTarget - output[a, targetC])


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
        # with torch.no_grad():
        a = torch.arange(output.size(0)).cuda()
        mask = torch.ones_like(output).cuda()
        mask[a, targetC] = 0
        maxNonTarget = torch.max(output * mask, dim=1)[0]
        return torch.mean(output[a, targetC] - maxNonTarget)

