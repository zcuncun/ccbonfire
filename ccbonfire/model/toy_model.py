#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:57:14
"""
toy_model.py
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    """  A example
    
    """
    def __init__(self, backbone, neck, head, loss, criterion):
        super(ToyModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
        self.criterion = criterion


    def forward(self, x) -> torch.Tensor:
        feature = self.backbone(x)
        feature = self.neck(feature)
        output = self.head(feature)
        return output

    

