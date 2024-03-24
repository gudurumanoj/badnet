"""
Generic helper functions
"""
import os
import torch
import numpy as np


def accuracy(output, target):
    """
    Calculate accuracy
    both output and target are torch tensors and are on cpu
    output already has maximization done
    """
    correct = output.eq(target.view_as(output)).sum().item()
    return correct / target.size(0)
   