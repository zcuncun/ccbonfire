#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:01:11
"""
toy_backbone.py
"""


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyBackbone(nn.Module):
    def __init__(self):
        super(ToyBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


def test():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    backbone = ToyBackbone()
    logging.debug(f"backbone ==> {backbone}")
    input_tensor = torch.zeros([6,3,12,12])
    logging.debug(f"input_tensor shape ==> {input_tensor.shape}")
    output_tensor = backbone(input_tensor)
    logging.debug(f"output_tensor shape ==> {output_tensor.shape}")


if __name__ == '__main__':
    test()
